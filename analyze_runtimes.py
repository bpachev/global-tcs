import os
import glob
import sys
import time
import json

def mtime_diff(later_file, base_file):
  if not os.path.exists(later_file) or not os.path.exists(base_file):
    return None
  
  diff = os.path.getmtime(later_file) - os.path.getmtime(base_file)
  # convert to hours
  return diff / 3600.

def get_times(jobdir):
  job_runtime = mtime_diff(f"{jobdir}/queuestate", f"{jobdir}/padcirc")
  if job_runtime is None: return None, []
  runtimes = []
  with open(jobdir+"/job.json") as fp: job_info = json.load(fp)
  jruns = job_info['jobRuns']  
  for run, rundir in zip(jruns, sorted(glob.glob(f"{jobdir}/runs/run*"))):
    basefile = f"{rundir}/padcirc.log"
    if not os.path.exists(basefile):
      basefile = f"{rundir}/PE0000/fort.15"
    runtime = mtime_diff(f"{rundir}/maxele.63.nc", basefile)
    outdir = run['outputs_dir']
    if runtime is None or not os.path.exists(outdir+"/fort.61.nc"):
      os.system(f"touch {outdir}/failed")
      bname = os.path.basename(rundir)
      
    runtimes.append(runtime)
  return job_runtime, runtimes

def main(jobdirs):
  run_times = []
  job_times = []
  for jobdir in jobdirs:
    if not os.path.exists(f"{jobdir}/padcirc"):
      print(f"Job {os.path.basename(jobdir)} not submitted, skipping")
      continue
    qstate = f"{jobdir}/queuestate"
    if not os.path.exists(qstate):
       continue #job never got to Pylauncher
    last_queuestate_update = os.path.getmtime(qstate)
    # see if job still running
    if (time.time()-last_queuestate_update) < 600:
      print(f"job {os.path.basename(jobdir)} still running")
      continue
    jtime, rtimes = get_times(jobdir)
    if jtime > 40:
      print(jobdir, "ran over 40 hours!")
    job_times.append(jtime)
    run_times.extend(rtimes)

  strange = [t for t in run_times if t is None]
  run_times = [t for t in run_times if t is not None]
  good = [t for t in run_times if t > 0]
  bad = [t for t in run_times if t <= 0]
  normal_jobs = [t for t in job_times if t is not None]
  bad_jobs = [t for t in job_times if t is None]
  print(f"Total runs {len(run_times)}. Good {len(good)}. Bad {len(bad)}. Strange {len(strange)}.")
  print(f"Average ADCIRC-only runtime for good runs {sum(good)/len(good)}. Max {max(good)}. Min {min(good)}.")
  print(f"Total completed jobs {len(normal_jobs)}. Total failed {len(bad_jobs)}.")
  print(f"Average job runtime {sum(normal_jobs)/len(normal_jobs)}. Max {max(normal_jobs)}. Min {min(normal_jobs)}.")

if __name__ == "__main__":
  jobdirs = sys.argv[1:]
  main(jobdirs)
