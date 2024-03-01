import numpy as np
import os
import h5py 
import netCDF4 as nc
from sklearn.neighbors import BallTree
from sklearn.metrics.pairwise import haversine_distances
import pandas as pd
import glob
from synthetic_storms import BASINS
import time
from multiprocessing import Pool
earth_radius = 6371

def partition_points(tracks, tree, radius=1e3):
   """Given a set of storm tracks and a set of points in a BallTree, partition the points among storms
   """

   query_radius = radius/earth_radius
   #all_inds = []
   #all_dists = []
   #all_track_nums = []
   start = time.time()
   N = len(tree.data)
   min_dist = np.full(N, np.inf)
   best_track = np.full(N, -1, np.int32)

   for i, df in enumerate(tracks):
     inds, dists = tree.query_radius(
                       np.deg2rad(df[['lat','lon']].values),
                       query_radius, return_distance=True
                   )
     for j in range(len(inds)):
       updates = min_dist[inds[j]] > dists[j]
       min_dist[inds[j][updates]] = dists[j][updates]
       best_track[inds[j][updates]] = i
     """dists = haversine_distances(tree.data, np.deg2rad(df[['lat','lon']].values))
     min_dists = np.min(dists, axis=1)
     inds = np.where(min_dists < 1e3)
     dists = min_dists[inds]
     all_inds.append(inds)
     all_dists.append(dists)
     all_track_nums.append(np.full(len(inds), i, dtype=int))"""
     #all_inds.extend(list(inds))
     #all_dists.extend(list(dists))
     #all_track_nums.append(np.full(sum([len(inds[j]) for j in range(len(inds))]), i, dtype=int))
  
   print("Query time", time.time()-start)
   #inds = np.concatenate(all_inds)
   #dists = np.concatenate(all_dists)
   #track_nums = np.concatenate(all_track_nums)
   # sort in order of increasing distance
   #order = np.argsort(dists)
   #inds, dists, track_nums = inds[order], dists[order], track_nums[order]
   # for each unique mesh node, determine the first time it occurs
   # the first occurrence corresponds to the closest distance to a track
   # which allows us to assign the node to the correct storm
   #_, first_occurrences = np.unique(inds, return_index=True)
   #inds, dists, track_nums = inds[first_occurrences], dists[first_occurrences], track_nums[first_occurrences]
   ind_list, dist_list = [], []
   for i in range(len(tracks)):
     mask = best_track == i
     ind_list.append(np.where(mask)[0])
     dist_list.append(earth_radius * min_dist[mask])

   return ind_list, dist_list


def unpack_storms(indir, outdir, mesh_tree, stat_tree, radius=1e3):
   """Given a run directory, unpack the storms into separate outputs
   """ 

   dfs = []
   landfalls = pd.read_csv(f"{indir}/landfalls.csv")
   for f in sorted(glob.glob(f"{indir}/track*csv")):
     dfs.append(pd.read_csv(f))

   mfile = f"{indir}/outputs/maxele.npz"
   sfile = f"{indir}/outputs/fort.61.nc"
   if not os.path.exists(mfile) or not os.path.exists(sfile):
     print(f"missing outputs for {indir}")
     return

   for i in range(len(dfs)):
     row = landfalls.iloc[i]
     basin = BASINS[int(row['basin'])]
     # get output directory
     dirname = f"{outdir}/{basin}/category{int(row['cat'])}/"
     storm_id = f"{int(row['year'])}{int(row['month']):02d}{int(row['tcnum']):02d}{int(row['tstep']):03d}"
     dirname += storm_id
     if os.path.exists(dirname+"/elevation.hdf5"):
       print("Skipping ", indir) 
       return
     if not i:
       mesh_ind_list, mesh_dist_list = partition_points(dfs, mesh_tree)
       stat_ind_list, stat_dist_list = partition_points(dfs, stat_tree)
       ark = np.load(f"{indir}/outputs/maxele.npz")
       station_ds = nc.Dataset(f"{indir}/outputs/fort.61.nc")
       station_zeta = station_ds["zeta"][:]

     #print(row, dfs[i].iloc[0])
     os.makedirs(dirname, exist_ok=True)
     dfs[i].to_csv(dirname+"/track.csv", index=False)
     with h5py.File(dirname+"/elevation.hdf5", "w") as ds:
        mesh_inds = mesh_ind_list[i]
        station_inds = stat_ind_list[i]
        ds['zeta_max'] = ark['zeta'][mesh_inds]
        ds['time_of_zeta_max'] = ark['zeta_time']
        ds['mesh_inds'] = mesh_inds
        ds['station_zeta'] = station_zeta[:, station_inds]
        ds['station_inds'] = station_inds
        ds['landfall_coord'] = np.array((row['lat'], row['lon']))
        ds['landfall_tstep'] = row['tstep']
     
     if os.path.exists(f"{indir}/tides.json"):
       os.system(f"cp {indir}/tides.json {dirname}")
     else:
       print(f"missing tides {indir}")
     
 
if __name__ == '__main__':
   import sys
   import pickle
   runsdir = sys.argv[1]
   outdir = sys.argv[2]

   if not os.path.exists('trees.pkl'):
     # need the file with mesh coordinates and station coordinates
     with h5py.File("global_mesh_points.hdf5") as ds:
       coords = np.deg2rad(np.column_stack([ds['lat'][:], ds['lon'][:]]))
       stat_coords = np.deg2rad(np.column_stack([ds['station_lat'][:], ds['station_lon'][:]]))
     print(f"ncoords {len(coords)}, nstat_coords {len(stat_coords)}")
   
     mesh_tree = BallTree(coords, metric='haversine')
     stat_tree = BallTree(stat_coords, metric='haversine')
     with open("trees.pkl", "wb") as fp: pickle.dump([mesh_tree,stat_tree], fp)
   else:
     with open("trees.pkl", "rb") as fp: mesh_tree, stat_tree = pickle.load(fp)

   dirs = sorted(glob.glob(runsdir+"/run*"))
   print("processing ", len(dirs), " runs")
   from functools import partial
   func = partial(unpack_storms, outdir=outdir, mesh_tree=mesh_tree, stat_tree=stat_tree)
   for d in dirs: func(d)
   #with Pool(32) as p:
   #  p.map(func, dirs) 
    
