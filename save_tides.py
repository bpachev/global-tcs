import os
import glob
import sys
import time
import json
import fix_tides

def save_tides(jobdir):
  with open(jobdir+"/job.json") as fp: job_info = json.load(fp)
  jruns = job_info['jobRuns']  
  for run, rundir in zip(jruns, sorted(glob.glob(f"{jobdir}/runs/run*"))):
    fort15 = f"{rundir}/fort.15"
    if not os.path.exists(fort15):
        print("missing", fort15)
        continue
    tidal_info = fix_tides.get_tides(fort15)
    outdir = run['outputs_dir']
    runid = outdir.split("/")[-2]
    corral_outdir = "/corral/utexas/musikal-project/global_tcs/"+runid
    if not os.path.exists(corral_outdir):
        print("missing", corral_outdir)
        continue
    with open(corral_outdir+"/tides.json", "w") as fp: json.dump(tidal_info, fp)  

def main(jobdirs):
  run_times = []
  job_times = []
  for jobdir in jobdirs:
    if not os.path.exists(f"{jobdir}/job.json"): continue
    print("Saving tides for", jobdir)
    save_tides(jobdir) 

if __name__ == "__main__":
  jobdirs = sys.argv[1:]
  main(jobdirs)
