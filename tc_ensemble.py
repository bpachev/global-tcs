from ch_sim import EnsembleSimulator
from ch_sim import adcirc_utils as au
import multiprocessing as mp
import winds
import os
import netCDF4 as nc
from functools import partial
import glob
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import json

"""
Launch point script for submitting heinously large collections of TC simulations.

@author Benjamin Pachev <benjaminpachev@utexas.edu>
"""

def setup_tc_run(run, run_dir, tide_fac_path=None):
    tidal_dir = run['tide']
    os.system(f"cp {tidal_dir}/fort.67.nc {run_dir}")
    # adjust fort15
    with nc.Dataset(run_dir + "/fort.67.nc") as ds:
        hotstart_days = ds["time"][0] / (24*3600)
        base_date_str = ds["time"].base_date.split("!")[0]
    fort15 = run_dir+"/fort.15"
    au.symlink_to_copy(fort15)
    wind_models = []
    rnday = 0
    for trackfile in sorted(glob.glob(run['tracks']+"/track*csv")):
        track = pd.read_csv(trackfile)
        rnday = max((len(track)-1)/8, rnday)
        wind_models.append(winds.HollandWinds(track))

    au.fix_fort_params(fort15, {"RND": rnday+hotstart_days, "BASE_DATE": base_date_str})
    base_date = datetime.strptime(base_date_str.strip(), "%Y-%m-%d %H:00:00 UTC")
    owi_start_date = winds.OWI_START_DATE + timedelta(days=hotstart_days)
    time = np.linspace(0, rnday*24*3600, int(rnday*24)+1)
    winds.make_owi_netcdf(run_dir, wind_models, time,
        start=owi_start_date, width=8, res=250)
    #avoid race conditions with shared tide_fac.out file
    os.chdir(run_dir)
    tides = au.compute_tides(rnday+hotstart_days, base_date, tide_fac_path)
    au.set_tides("fort.15", tides)
    with open(run['tracks']+'/tides.json', 'w') as fp:
        json.dump(tides, fp)

class TCEnsemble(EnsembleSimulator):

    def __init__(self, **kwargs):
        super().__init__(
           deps=['winds.py', 'downsample_tc_output.py', 'fix_tides.py'],
         **kwargs)


    def add_commandline_args(self, parser):
        parser.add_argument("--runs", required=True)
        parser.add_argument("--tides", required=True)
        parser.add_argument("--cache", required=True)
        parser.add_argument("--inputs-dir", required=True)
        parser.add_argument("--execs-dir", required=True)
        parser.add_argument("--runtime", default=4, type=float)
        parser.add_argument("--nodes", default=10, type=int)
        parser.add_argument("--maxJobNodes", default=100, type=int)
        parser.add_argument("--queue", default="normal")

        parser.add_argument("--retry", required=False, action="store_true")

    def setup_job(self):
        super().setup_job()
        tide_fac_exec = self.config['execs_dir']+"/tide_fac"
        runs = list(zip(self.job_config["jobRuns"], self.run_dirs))
        # setup in parallel
        func = partial(setup_tc_run, tide_fac_path=tide_fac_exec)
        with mp.Pool(min(32, len(runs))) as pl:
            pl.starmap(func, runs)

        os.system(f"cp {self.get_arg('cache')} {self.job_config['job_dir']}/cache.zip")

    def make_preprocess_command(self, run, run_dir):
        job_dir = self.job_config['job_dir']
        adcprep_cache = job_dir+"/cache.zip"
        # copy zip archive, unpack cache, localize the fort.15 file
        #writers, workers = self.get_writers_and_workers()
        new_rndy = au.snatch_fort_params(run_dir+"/fort.15", ["RND"])["RND"]
        return (
                    f"unzip -d {run_dir} {adcprep_cache} > /dev/null;"+
                    f"sed -i  's/.*RND.*/{new_rndy}      ! RNDY /g' {run_dir}/PE*/fort.15;"+
                    f"python3 {job_dir}/fix_tides.py {run_dir}"                    
#f"cd {run_dir};"+
                    #f"printf '{workers}\\n4\\nfort.14\\nfort.15\\n' | {job_dir}/adcprep > {run_dir}/adcprep.log"
                    #+ f";cd {job_dir}"
            )

    def make_postprocess_command(self, run, run_dir):
        cmd = super().make_postprocess_command(run, run_dir)
        return (
          cmd + f"; python3 {self.job_config['job_dir']}/downsample_tc_output.py {run['outputs_dir']} {run_dir}"
        )


if __name__ == "__main__":

    sim = TCEnsemble(allocation="DMS23001")
    args = sim.args
    if args.action == "setup":
       for path in [args.runs, args.inputs_dir, args.execs_dir, args.tides, args.cache]:
         if not os.path.exists(path): raise RuntimeError(f"No such file or directory {path}")
       config = {
          "inputs_dir": args.inputs_dir,
          "node_count": args.nodes,
          "runtime": args.runtime,
          "queue": args.queue,
          "execs_dir": args.execs_dir,
          "maxJobNodes": args.maxJobNodes
       }
       runsdir = args.runs
       # include only the runs with actual tides
       tides = sorted(glob.glob(args.tides+"/20*/fort.67.nc"))
       runs = []
       retry = args.retry is True
       for i, dirname in enumerate(sorted(glob.glob(runsdir+"/run*"))):
          run = {
                  "tracks": dirname,
                  "tide": tides[i%len(tides)].replace("/fort.67.nc", ""),
                  "outputs_dir": dirname+"/outputs",
                  "output_files": ["fort.61.nc"]
          }
          if retry:
            rfile = dirname+"/retrying"
            if os.path.exists(rfile): continue
            f61 = dirname+"/outputs/fort.61.nc"
            if os.path.exists(f61):
                if os.path.getsize(f61) > 5e5: continue
            os.system(f"touch {rfile}")

          runs.append(run)
       print(runs[:1], runs[-1:])

    else:
        runs = []
        config = {}

    sim.run(
        runs=runs,
        maxJobRuntime=48,
        processors_per_node=50,
        no_writers=True,
        **config
    )
