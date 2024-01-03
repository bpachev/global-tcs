import pandas as pd
from config import IBTRACKS_DIR
import glob
import os
# import nederhoff
import numpy as np
from datetime import datetime, timedelta

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


    def select_storms(self, landfalls="landfall.csv", bucket_size=25, min_cat=2):
        """Select storms across basins
        """

        df = pd.read_csv(landfalls)
        res = df[df["cat"]>=min_cat].groupby(["basin", "cat"]).head(bucket_size)
        return res


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
            track = gb[key]
            print(track)
            dirname = outdir+f"/{basin}_{year}_{num}"
            os.makedirs(dirname, exist_ok=True)
            fname = f'{dirname}/fort.22'
            break
            #self.write_adcirc_input(track, fname)



    def write_adcirc_input(self, trackdata: pd.DataFrame, outfile):
        """Given the description of a tc track, create a fort.22 file
        """
        trackdata['tstep'] -= trackdata['tstep'].min()
        self.basin = BASINS[int(trackdata.iloc[0]['basin'])]
        fields = "BASIN,CY,YYYYMMDDHH,TECHNUM/MIN,TECH,TAU,LatN/S,LonE/W,VMAX,MSLP,TY,RAD,WINDCODE,RAD1,RAD2,RAD3,RAD4,RADP,RRP,MRD,GUSTS,EYE,SUBREGION,MAXSEAS,INITIALS,DIR,SPEED,STORMNAME,DEPTH,SEAS".split(",")
        lengths = {'BASIN': 2, 'CY': 3, 'YYYYMMDDHH': 11, 'TECHNUM/MIN': 3, 'TECH': 5, 'TAU': 4, 'LatN/S': 5, 'LonE/W': 6, 'VMAX': 4, 'MSLP': 5, 'TY': 3, 'RAD': 4, 'WINDCODE': 4, 'RAD1': 5, 'RAD2': 5, 'RAD3': 5, 'RAD4': 5, 'RADP': 5, 'RRP': 5, 'MRD': 4, 'GUSTS': 4, 'EYE': 4, 'SUBREGION': 4, 'MAXSEAS': 4, 'INITIALS': 4, 'DIR': 4, 'SPEED': 4, 'STORMNAME': 11, 'DEPTH': 2, 'SEAS': 2}
        default_values = {
            "BASIN": self.basin,
            "CY": " 00",
            "YYYYMMDDHH": " 2030010100",
            "WINDCODE": "AAA", # symmetric vortex
            "STORMNAME": "INVEST",
            "TECH": "BEST", # use best track
            "RAD2": 0,
            "RAD3": 0,
            "RAD4": 0,
            "SUBREGION": "L",
        }

        start_time = datetime(year=2030, day=1, month=1)
        for f in fields:
            if f not in default_values:
                default_values[f] = ' ' * lengths[f]
       
        #_ , dr35 = nederhoff.wind_radii_nederhoff(trackdata['vmax'], trackdata['lat'])
        # the Holland model doesn't need r35
        trackdata['r35'] = trackdata['rmax'] + 50
        trackdata.to_csv(outfile.replace(".22",".csv"), index=False)
        with open(outfile, 'w') as fp:
            for idx, row in trackdata.iterrows():
                # we can't use Pandas because of the fixed-width format
                # this is NWS=8 format
                vals = {**default_values}
                r35 = row['r35']
                #vals["TAU"] = int(3*row['tstep'])
                vals["TAU"] = 0
                time = start_time + timedelta(seconds=3*3600*row['tstep'])
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
                line = ",".join([vals[field] for field in fields])
                fp.write(line+"\n")

if __name__ == "__main__":
    tcs = SyntheticTCs()
    trackdata_path = "/work2/09631/maxzhao88/frontera/data_generation/rand_storm_1_modified.csv"  
    trackdata_df = pd.read_csv(trackdata_path)
    output_path = "/work2/09631/maxzhao88/frontera/parametric_inputs/fort.22" 
    tcs.write_adcirc_input(trackdata_df, output_path)
