import pandas as pd
from config import IBTRACKS_DIR
import glob
import os
# import nederhoff
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
from sklearn.metrics.pairwise import haversine_distances
from itertools import zip_longest
import winds

"""
Contains code for subsetting synthetic tropical cyclone (TC) data and outputting ADCIRC input files.

@author Benjamin Pachev <benjamin.pachev@gmail.com>
"""

KNOTS_TO_KMH = 1.852
MS_TO_KNOTS = 1.94384
KMH_TO_KNOTS = 1/KNOTS_TO_KMH
KM_TO_NM = 0.539957
ONE_TO_TEN = 0.8928  
TEN_TO_ONE = 1./ONE_TO_TEN

BASINS = ["EP", "NA", "NI", "SI", "SP", "WP"]

def consolidate_ibtracs():
    dfs = []
    for basin in BASINS:
        files = files = sorted(glob.glob(IBTRACKS_DIR+f"/STORM_DATA_IBTRACS_{basin}*txt"))
        names = [
                'year', 'month', 'tcnum', 'tstep',
                'basin', 'lat', 'lon', 'minpres',
                'vmax', 'rmax', 'cat', 'landfall', 'ldist'
        ]

        for i,fname in enumerate(files):
            print(basin, i)
            df = pd.read_csv(fname, names=names, header=None)
            df['year'] += 1000*i
            dfs.append(df)

        
    tracks = pd.concat(dfs)
    tracks.to_csv("ibtracs_consolidated.csv", index=False)


class SyntheticTCs:
    """Represents a collection of synthetic TCs
    """

    def __init__(self, fname="ibtracs_consolidated.csv"):
        """Load in all storms
        """


        self.tracks = pd.read_csv(fname)

    def analyze_landfall(self, outfile="landfall.csv",  min_cat=1):
        """Analyze the distribution of landfall events
        
        Note we consider only the storm category at landfall
        """

        # augment tracks dataframe with additional fields
        self.tracks["prev_landfall"] = self.tracks["landfall"].shift(1)
        df = self.tracks
        mask = ((df["prev_landfall"]==0) & (df["landfall"] == 1) & (df["cat"] >= min_cat))
        rows = df[mask]
        print("landfall events by category")
        print(rows["cat"].value_counts())
        rows.to_csv(outfile, index=False)

    def analyze_bypassing(self, min_cat=1, max_dist=500):
        """Analyze bypassing storms
        """

        df = self.tracks
        df = df[(df['cat'] >= min_cat) & (df['ldist'] <= max_dist)]
        #bypassing = df.groupby(['basin', 'year', 'tcnum'])['ldist'].min()
        #bypassing = bypassing[bypassing > 0]
        # what we ultimately care about for bypassing storms is the effective category
        # which is defined as the max wind experienced on land
        # we need a way of estimating that. . . .
        DP = (winds.pback - df['minpres']) * 100
        DP[DP<100] = 100
        vmax = df["vmax"] / (winds.BLAdj * winds.one2ten)
        holland_b = (winds.rhoa * np.exp(1)/DP) * (vmax) ** 2
        holland_b = holland_b.clip(1, 2.5)
        r_denom = df['ldist'].clip(lower=df['rmax'])
        r_non_dimensional = (df['rmax'] / r_denom) ** holland_b
        radial_velocity = (r_non_dimensional * np.exp(1-r_non_dimensional)) ** .5 * vmax
        radial_velocity *= (winds.BLAdj * winds.one2ten)
        df['vland'] = radial_velocity
        cutoff_vland = df['vmax'].min()
        print("Cutoff vland", cutoff_vland)
        df = df[df['vland'] >= cutoff_vland]
        gb = df.groupby(['basin', 'year', 'tcnum'])
        bypassing = gb[['ldist']].min()
        bypassing['vland'] = gb['vland'].max()
        bypassing['vland_idxmax'] = gb['vland'].idxmax()
        bypassing['count'] = gb['vland'].count()
        bypassing = bypassing[bypassing['ldist'] > 0]
        print(bypassing)
        print(bypassing.reset_index()['basin'].value_counts())
        print(bypassing['count'].describe())
        pseudolandfalls = df.loc[bypassing['vland_idxmax']]
        pseudolandfalls.to_csv('pseudolandfalls.csv', index=False)
        return pseudolandfalls

    def select_storms(self, landfalls="landfall.csv", bucket_size=25, min_cat=2, max_cat=5):
        """Select storms across basins
        """

        df = pd.read_csv(landfalls)
        res = df[(df["cat"]>=min_cat)&(df["cat"]<=max_cat)].groupby(["basin", "cat"]).head(bucket_size)
        return res

    def separation(self, track1, track2):
        """Determine the minimum separation in km between tracks
        """

        dists = haversine_distances(
            np.deg2rad(track1[['lat', 'lon']].values),
            np.deg2rad(track2[['lat', 'lon']].values)
        )
        return np.min(dists) * 6371

    def write_packed_inputs(self, outdir, landfalls,
            days_before=4, days_after=.5, max_basin_storms=1,
            min_separation=1e3
            ):
        """Group storms into runs
        """
        n = len(landfalls)
        # step 1 - extract dataframes
        os.makedirs(outdir, exist_ok=True)
        gb = self.tracks.groupby(["basin", "year", "tcnum"])
        basin_storm_inds = defaultdict(list)
        storm_tracks = []
        i = 0
        for idx, row in landfalls.iterrows():
            key = (row["basin"], row["year"], row["tcnum"])
            basin, year, num = map(int, key)
            track = self.tracks.iloc[gb.indices.get(key)]
            start = max(0, int(row['tstep']-days_before*8))
            stop = min(len(track), int(row['tstep'] + days_after*8)+1)
            basin_storm_inds[basin].append(i)
            storm_tracks.append(track.iloc[start:stop])
            i += 1
 
        # step 2 - try to pack compatible storms together
        # uses greedy algorithm
        basin_storm_groups = {}
        for basin, inds in basin_storm_inds.items():
            groups = []
            print("Basin", basin)
            taken = set()
            for i in range(len(inds)):
                if i in taken: continue
                curr = [inds[i]]
                tot_bad = 0
                for j in range(i+1, len(inds)):
                    if len(curr) >= max_basin_storms: break 
                    if j in taken: continue
                    cand_ind = inds[j]
                    cand_track = storm_tracks[cand_ind]
                    good = True
                    for k in curr:
                        sep = self.separation(cand_track, storm_tracks[k])
                        if sep < min_separation:
                            good = False
                            tot_bad += 1
                            break
                    if good:
                        curr.append(cand_ind)
                        taken.add(j)
                    if tot_bad > 20: break
                groups.append(curr)

            assert sum([len(g) for g in groups]) == len(inds)
            basin_storm_groups[basin] = groups
            print(len(inds), len(groups), len(inds)/len(groups))

        # step 3 - create outputs
        all_groups = [groups for basin, groups in basin_storm_groups.items()]
        total_runs = max(map(len, all_groups))
        ndigits = len(str(total_runs))
        runno = 0
        for storms_list in zip_longest(*all_groups):
            rundir = outdir+"/run"+str(runno).zfill(ndigits)
            os.makedirs(rundir, exist_ok=True)
            runno += 1
            stormno = 0
            inds = []
            for storms in storms_list:
                if storms is None: continue
                for ind in storms:
                    track = storm_tracks[ind]
                    track.to_csv(rundir+f"/track{stormno:02d}.csv", index=False)
                    stormno += 1
                    inds.append(ind)
            landfalls.iloc[inds].to_csv(rundir+f"/landfalls.csv")




    def write_adcirc_inputs(self,
            outdir,
            landfalls,
            days_before=4,
            days_after=.5,
       ):
        os.makedirs(outdir, exist_ok=True)
        gb = self.tracks.groupby(["basin", "year", "tcnum"])
        for idx, row in landfalls.iterrows():
            key = (row["basin"], row["year"], row["tcnum"])
            basin, year, num = map(int, key)
            track = self.tracks.iloc[gb.indices.get(key)]
            print(track)
            dirname = outdir+f"/{basin}_{year}_{num}"
            os.makedirs(dirname, exist_ok=True)
            fname = f'{dirname}/fort.22'
            break
            #self.write_adcirc_input(track, fname)



def write_adcirc_input(
    trackdata: pd.DataFrame,
    outfile,
    start_time=datetime(year=2030, day=1, month=1),
    spinup_days=0,
    ):
    """Given the description of a tc track, create a fort.22 file
    """
    trackdata['tstep'] -= trackdata['tstep'].min()
    basin = BASINS[int(trackdata.iloc[0]['basin'])]
    fields = "BASIN,CY,YYYYMMDDHH,TECHNUM/MIN,TECH,TAU,LatN/S,LonE/W,VMAX,MSLP,TY,RAD,WINDCODE,RAD1,RAD2,RAD3,RAD4,RADP,RRP,MRD,GUSTS,EYE,SUBREGION,MAXSEAS,INITIALS,DIR,SPEED,STORMNAME,DEPTH,SEAS".split(",")
    lengths = {'BASIN': 2, 'CY': 3, 'YYYYMMDDHH': 11, 'TECHNUM/MIN': 3, 'TECH': 5, 'TAU': 4, 'LatN/S': 5, 'LonE/W': 6, 'VMAX': 4, 'MSLP': 5, 'TY': 3, 'RAD': 4, 'WINDCODE': 4, 'RAD1': 5, 'RAD2': 5, 'RAD3': 5, 'RAD4': 5, 'RADP': 5, 'RRP': 5, 'MRD': 4, 'GUSTS': 4, 'EYE': 4, 'SUBREGION': 4, 'MAXSEAS': 4, 'INITIALS': 4, 'DIR': 4, 'SPEED': 4, 'STORMNAME': 11, 'DEPTH': 2, 'SEAS': 2}
    default_values = {
        "BASIN": basin,
        "CY": " 00",
        "YYYYMMDDHH": " 2030010700",
        "WINDCODE": "AAA", # symmetric vortex
        "STORMNAME": "INVEST",
        "TECH": "BEST", # use best track
        "RAD2": 0,
        "RAD3": 0,
        "RAD4": 0,
        "SUBREGION": "L",
    }

    for f in fields:
        if f not in default_values:
            default_values[f] = ' ' * lengths[f]
    #_ , dr35 = nederhoff.wind_radii_nederhoff(trackdata['vmax'], trackdata['lat'])
    # the Holland model doesn't need r35
    trackdata['r35'] = trackdata['rmax'] + 50
    trackdata.to_csv(outfile.replace(".22",".csv"), index=False)
    with open(outfile, 'w') as fp:
        prepended = False
        spinup_delta = spinup_days*24*3600
        for idx, row in trackdata.iterrows():
            # we can't use Pandas because of the fixed-width format
            # this is NWS=8 format
            vals = {**default_values}
            r35 = row['r35']
            #vals["TAU"] = int(3*row['tstep'])
            vals["TAU"] = 0
            time = start_time + timedelta(seconds=spinup_delta+3*3600*row['tstep'])
            vals["LatN/S"] = f"{int(10*row['lat'])}N" 
            vals["LonE/W"] = f"{int(10*(360-row['lon']))}W"
            # multiply by a factor of 1/.88 to convert from
            # ten-minute winds to one-minute winds
            # ADCIRC expects one-minute winds as input
            vals["VMAX"] = f"{round(row['vmax']*MS_TO_KNOTS*TEN_TO_ONE)}"
            vals["MRD"] = f"{round(row['rmax']*KM_TO_NM)}"
            vals["YYYYMMDDHH"] = time.strftime("%Y%m%d%H")
            
            if np.isnan(r35):
                vals["RAD"] = 0
                vals["RAD1"] = 0
            else:
                vals["RAD"] = 34 # we provide information on winds of 35/34 knots
                vals["RAD1"] = f"{round(row['r35']*KM_TO_NM)}"
            
            vals["MSLP"] = f"{round(row['minpres'])}"
            vals["RADP"] = 1013 # background pressure
            vals["RRP"] = 250  # radius to background pressure
            for k in vals: vals[k] = str(vals[k]).rjust(lengths[k])
            if not prepended:
              for delta in range(0, spinup_delta, 3*3600):
                 time = start_time + timedelta(seconds=delta)
                 prepend_vals = {**vals}
                 prepend_vals["YYYYMMDDHH"] = time.strftime("%Y%m%d%H").rjust(lengths["YYYYMMDDHH"])
                 fp.write(','.join([prepend_vals[field] for field in fields]) + '\n')
              prepended=True
                 

            line = ",".join([vals[field] for field in fields])
            fp.write(line+"\n")


if __name__ == "__main__":
    tcs = SyntheticTCs()
    pseudolandfalls = tcs.analyze_bypassing(min_cat=2)
    selected = pseudolandfalls.groupby('basin').head(8000)
    tcs.write_packed_inputs("full_bypassing_inputs_above_cat2", selected, max_basin_storms=4)
    landfalls = tcs.select_storms(bucket_size=5000, min_cat=1, max_cat=1)
    #print(landfalls)
    tcs.write_packed_inputs("cat1_packed_inputs", landfalls, max_basin_storms=4)
    #tcs.write_adcirc_inputs("holland_inputs", landfalls)
