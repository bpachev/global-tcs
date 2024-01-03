import numpy as np
import pandas as pd
from haversine import haversine_vector
import math
import netCDF4 as nc
from datetime import datetime
import argparse

"""A collection of utilities for working with ADCIRC wind files
"""

# -------------------------------------------------------------------------
# Const
# -------------------------------------------------------------------------
#nm2m=1852. # 1 nautical mile to meters
#kt2ms=nm2m/3600.  # knots to m/s
omega=2*np.pi/(3600.*24. + .2) # angular speed omega=2pi*f(=frequency of earth : 1 cycle per day) 2pi* 1 / day in seconds
#rhoa=1.15 #air density  Kg/m^3
rhoa = 1.293
rhowat0 = 1e3
G = 9.80665
radius=6378388 #137. # earth's radius according to WGS 84
deg2m=np.pi*radius/180.  # ds on cicle equals ds=r*dth - dth=pi/180
one2ten=0.8928  # coefficient for going from 1m to 10m in velocities
BLAdj=0.9
pback = 1013
OWI_START_DATE=datetime(2030, 1, 1)


class HollandWinds:
    """A Python class for generating winds with the symmetric Holland model
    """

    def __init__(self, track, dt=3*3600):
        """Initialize the wind model

        Parameters
        ----------
        track - pd.DataFrame
            A Pandas DataFrame with best track information.
            Must contain the columns: lat, lon, vmax, rmax, minpres
        dt - timedelta in seconds 
        """

        self.lat = track["lat"].values
        self.lon = track["lon"].values
        self.lon[self.lon > 180] -= 360
        # convert vmax from 10-minute to 1-minute
        self.vmax = track["vmax"].values / one2ten
        self.rmax = track["rmax"].values
        self.minpres = track["minpres"].values
        self.dt = dt
        self.n = len(track)
        # compute translational velocities
        self._compute_translational_vel()

    def _compute_translational_vel(self):
        x = self.lon
        y= self.lat
        # convert timestep to seconds
        dt = self.dt
    
        velocity = np.zeros((len(x), 2))
        velocity[1:, 0] = np.cos(np.deg2rad((y[1:]-y[:-1])/2)) * (x[1:]-x[:-1]) / dt
        velocity[1:, 1] = (y[1:]-y[:-1]) / dt
        velocity[0] = velocity[1]
        self.vtrx = velocity[:,0] * deg2m  #adjust for latitude
        self.vtry = velocity[:,1] * deg2m    
        self.vtr = np.sqrt(self.vtrx**2+self.vtry**2)

    def evaluate(self, t, lats, lons):
        """Evaluate the model at a given time and set of coordinates

        Parameters
        ----------
        t - time in hours
        lats - latitudes to evaluate at
        lons - longitudes to evaluate at
        
        Returns
        ----------
        windx, windy, pres
        """
        # do interpolation
        ind = t/self.dt
        if ind < 0 or ind >= self.n:
            # out of bounds - zero winds and background pressure
            return np.zeros_like(lats), np.zeros_like(lats), np.full_like(lats, pback)

        last = int(math.floor(ind))
        curr = int(math.ceil(ind))
        lam = ind-last
        vmax = (1-lam) * self.vmax[last] + lam * self.vmax[curr]
        rmaxh = (1-lam) * self.rmax[last] + lam * self.rmax[curr]
        lat0 = (1-lam) * self.lat[last] + lam * self.lat[curr]
        lon0 = (1-lam) * self.lon[last] + lam * self.lon[curr]
        pc = (1-lam) * self.minpres[last] + lam * self.minpres[curr]
        vtx = (1-lam) * self.vtrx[last] + lam * self.vtrx[curr]
        vty = (1-lam) * self.vtry[last] + lam * self.vtry[curr]
        lons = lons.copy()
        lons[lons > 180] -= 360
        lons[lons < -180] += 360
        if lon0 > 180: lon0 -= 360
        elif lon0 < -180: lon0 += 360
        coors = [x for x in zip(lats,lons)]
        r = haversine_vector(coors, len(coors)*[(lat0, lon0)]) * 1000
        r[r<1e-3] = 1e-3
        # pressure deficit
        DP = (pback-pc)*100
        DP = max(DP, 100)
        vtr = (vtx**2 + vty**2) ** .5
        # need to subtract the translational speed from the vmax
        vmax -= vtr
        vmax /= BLAdj
        bh = (rhoa*np.exp(1)/DP)*(vmax)**2
        # cap bh to be within proper ranges
        bh = min(2.5, max(bh, 1))
        theta=np.arctan2(np.deg2rad(lats-lat0),np.deg2rad(lons-lon0))
        fcor = 2*omega*np.sin(np.deg2rad(lats)) #coriolis force
        r_nd = (rmaxh*1e3/r)**bh
        ur = (
            r_nd * np.exp(1-r_nd) * vmax**2
            + (r*fcor/2)**2
        )**0.5 - r*abs(fcor)/2
        pres_prof = pc+DP*np.exp(-r_nd**bh)/100
        ux = -ur*np.sin(theta) * BLAdj * one2ten
        uy = ur*np.cos(theta) * BLAdj * one2ten
        mul = np.abs(ur)/vmax
        return ux + mul * vtx, uy + mul * vty, pres_prof

    def get_max_time(self):
        return (self.n - 1) * self.dt

    def compute_track(self, time):
        """Get track at given timepoints
        """

        inds = time / self.dt
        last = np.floor(inds).astype(int)
        curr = np.ceil(inds).astype(int)
        lam = inds-last
        bad = (last < 0) | (curr >= self.n)
        last[bad] = 0
        curr[bad] = self.n-1
        lat = (1-lam) * self.lat[last] + lam * self.lat[curr]
        lat[bad] = np.nan
        lon = (1-lam) * self.lon[last] + lam * self.lon[curr]
        lon[bad] = np.nan

        return lat, lon


def create_dummy_group(ds, start, time):
    """Create a dummy background wind field group
    """

    g = ds.createGroup("Main")
    g.rank = 1
    dummy_lats = np.array([-90, 0, 90])
    dummy_lons = np.array([-179.99, 0, 179.99])
    g.createDimension("time", len(time))
    timevar = g.createVariable("time", "i8", ("time",))
    timevar[:] = (time/60).astype(np.int64)
    timevar.units = "minutes since " + start.strftime("%Y-%m-%d %H:%M:%S")
    g.createDimension("xi", 3)
    g.createDimension("yi", len(dummy_lats))
    dims = ("time", "yi", "xi")
    lat = g.createVariable("lat", "f8", dims)
    lon = g.createVariable("lon", "f8", dims)
    lat[:] = dummy_lats[np.newaxis, np.newaxis, :] * np.ones(len(dummy_lons))[np.newaxis, :, np.newaxis]
    lon[:] = dummy_lons[np.newaxis, :, np.newaxis] * np.ones(len(dummy_lats))[np.newaxis, np.newaxis, :]
    u10 = g.createVariable("U10", "f8", dims)
    v10 = g.createVariable("V10", "f8", dims)
    pres = g.createVariable("PSFC", "f8", dims)
    u10[:] = 0.0
    v10[:] = 0.0
    pres[:] = 1013
    u10.units = "m s-1"
    v10.units = "m s-1"
    pres.units = "mb"


def make_tiered_offsets(base=3, ntiers=3, res=200):
    pass

def make_owi_netcdf(outdir, wind_models, time, width=3, res=225, start=OWI_START_DATE):
    """Create a multi-storm OWI NetCDF ADCIRC input
    """

    outfile = outdir+"/fort.22.nc"
    # TODO come up with a smarter way of adjusting
    # the grid resolution
    offsets = np.linspace(-width/2, width/2, res)
    lat_offsets = offsets[:, np.newaxis] * np.ones(res)[np.newaxis, :]
    lon_offsets = offsets[np.newaxis, :] * np.ones(res)[:, np.newaxis]
    with nc.Dataset(outfile, "w") as ds:
        grps = ["Main"]
        create_dummy_group(ds, start, time)
        for i, wind_model in enumerate(wind_models):
            rank = i+2
            name = f"Storm{i}"
            grps.append(name)
            g = ds.createGroup(name)
            g.rank = rank
            # determine valid times
            valid_time = time[time<=wind_model.get_max_time()]
            g.createDimension("time", len(valid_time))
            g.createDimension("yi", res)
            g.createDimension("xi", res)

            latc, lonc = wind_model.compute_track(valid_time)
            # generate the lat/lon grids
            lat = latc[:, np.newaxis, np.newaxis] + lat_offsets[np.newaxis, :]
            lon = lonc[:, np.newaxis, np.newaxis] + lon_offsets[np.newaxis, :]
            dims = ("time", "yi", "xi")
            latvar = g.createVariable("lat", "f8", dims)
            lonvar = g.createVariable("lon", "f8", dims)
            latvar[:] = lat
            lonvar[:] = lon
            lonvar.units = "degrees_east"
            latvar.units = "degrees_north"

            u10 = np.zeros_like(lat)
            v10 = np.zeros_like(u10)
            pres = np.zeros_like(u10)
            for j, t in enumerate(valid_time):
                profile = wind_model.evaluate(t, lat[j].flatten(), lon[j].flatten())
                shp = u10[j].shape
                u10[j] = profile[0].reshape(shp)
                v10[j] = profile[1].reshape(shp)
                pres[j] = profile[2].reshape(shp)

            # create variables
            u10var = g.createVariable("U10", "f8", dims)
            v10var = g.createVariable("V10", "f8", dims)
            presvar = g.createVariable("PSFC", "f8", dims)
            u10var.units = v10var.units = "m s-1"
            presvar.units = "mb"
            u10var[:] = u10
            v10var[:] = v10
            presvar[:] = pres
            timevar = g.createVariable("time", "i8", ("time",))
            timevar[:] = (valid_time/60).astype(np.int64)
            timevar.units = "minutes since " + start.strftime("%Y-%m-%d %H:%M:%S")

        ds.group_order = " ".join(grps)
        ds.conventions = "CF-1.6 OWI-NWS13"

if __name__ == "__main__":

    # Setup argument parser
    parser = argparse.ArgumentParser(description='Generate OWI NetCDF ADCIRC input')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument('--width', type=int, default=3, help='Width parameter')
    parser.add_argument('--res', type=int, default=225, help='Resolution parameter')
    args = parser.parse_args()

    models = []
    csv_file_path = "/work2/09631/maxzhao88/frontera/data_generation/rand_storm_1_modified.csv"
    df = pd.read_csv(csv_file_path)
    rows_per_storm = 41
    number_of_storms = len(df) // rows_per_storm
    # print(number_of_storms)
    for storm_index in range(number_of_storms):
        storm_data = df.iloc[storm_index * rows_per_storm : (storm_index + 1) * rows_per_storm]
        models.append(HollandWinds(storm_data))

    make_owi_netcdf(args.outdir, models, np.linspace(0, 3600*24*5, 24*5+1), width=args.width, res=args.res)
    
