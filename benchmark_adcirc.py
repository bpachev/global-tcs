from ch_sim import BaseSimulator, EnsembleSimulator
import os
import glob
import netCDF4 as nc
import ch_sim.adcirc_utils as au
from datetime import datetime

"""class BenchmarkSim(BaseSimulator):

    def add_commandline_args(self, parser):
        parser.add_argument("--cache", required=False)

    def make_preprocess_command(self, run, run_dir):
        job_dir = self.job_config['job_dir']
        adcprep_cache = self.get_arg("cache")
        if adcprep_cache is None:
            # invoke adcprep as usual
            return super().make_preprocess_command(run, run_dir)
        else:
            # unpack cache, localize the fort.15 file
            writers, workers = self.get_writers_and_workers()
            return (
                    f"unzip -d {run_dir} {adcprep_cache} > /dev/null;"+
                    f"printf '{workers}\\n4\\nfort.14\\nfort.15\\n' | {job_dir}/adcprep > {run_dir}/adcprep.log"
            )"""

class UnpackedSim(EnsembleSimulator):

    def __init__(self, **kwargs):
        super().__init__(
                deps = ['downsample_tc_output.py', 'fix_tides.py'], **kwargs)

    def add_commandline_args(self, parser):
        parser.add_argument("--cache", required=True)
        parser.add_argument("--packed-dir", required=True)
        parser.add_argument("--tide", required=True)

    def make_preprocess_command(self, run, run_dir):
        job_dir = self.job_config['job_dir']
        adcprep_cache = job_dir+"/cache.zip"
        # copy zip archive, unpack cache, localize the fort.15 file
        #writers, workers = self.get_writers_and_workers()
        new_rndy = au.snatch_fort_params(run_dir+"/fort.15", ["RND"])["RND"]
        return (
                    f"unzip -d {run_dir} {adcprep_cache} > /dev/null;"+
                    f"sed -i  's/.*RND.*/{new_rndy}      ! RNDY /g' {run_dir}/PE*/fort.15;"+
                    f"sed -i  's/.*NWS.*/8          ! NWS /g' {run_dir}/PE*/fort.15;"+
                    f"sed -i 's/3600 3600/2030 01 01 00 1 0.9/g' {run_dir}/PE*/fort.15;"+
                    f"python3 {job_dir}/fix_tides.py {run_dir}")

    def setup_job(self):
        super().setup_job()
        tidal_dir = self.get_arg('tide')
        for run_dir in self.run_dirs:
            os.system(f"cp {tidal_dir}/fort.67.nc {run_dir}")
        max_rnday = max([run['rnday'] for run in self.job_config['jobRuns']])
        # use first run directory to modify shared fort.15
        run_dir = self.run_dirs[0]
        # adjust fort15
        with nc.Dataset(run_dir + "/fort.67.nc") as ds:
            hotstart_days = ds["time"][0] / (24*3600)
            base_date_str = ds["time"].base_date.split("!")[0]
        base_date = datetime.strptime(base_date_str.strip(), "%Y-%m-%d %H:00:00 UTC")
        tides = au.compute_tides(max_rnday+hotstart_days, base_date, self.config['execs_dir']+"/tide_fac")
        for run, run_dir in zip(self.job_config['jobRuns'], self.run_dirs):
          fort15 = run_dir+"/fort.15"
          au.fix_fort_params(fort15, {"BASE_DATE": base_date_str, 'RND': run['rnday']+hotstart_days})
          au.set_tides(fort15, tides)
        self.hotstart_days = hotstart_days
        os.system(f"cp {self.get_arg('cache')} {self.job_config['job_dir']}/cache.zip")

    def make_postprocess_command(self, run, run_dir):
        cmd = super().make_postprocess_command(run, run_dir)
        return (
          cmd + f"; python3 {self.job_config['job_dir']}/downsample_tc_output.py {run['outputs_dir']} {run_dir}"
        )


def packed_inputs_to_unpacked_inputs(dirname):
    """Given a directory with packed storm inputs, set up a set of NWS=8 runs
    """

    from synthetic_storms import write_adcirc_input
    import pandas as pd

    trackfiles = sorted(glob.glob(dirname+"/track*csv"))
    inputdirs = [dirname+f'/unpacked_inputs/storm{i:02d}' for i in range(len(trackfiles))]
    rndays = []
    for i, indir in enumerate(inputdirs):
        df = pd.read_csv(trackfiles[i])
        os.makedirs(indir, exist_ok=True)
        write_adcirc_input(df, indir+'/fort.22', start_time=datetime(year=2030, month=1, day=1), spinup_days=7)
        rndays.append((len(df)-1)/8)
    return inputdirs, rndays 

if __name__ == "__main__":
  
  sim = UnpackedSim(allocation="DMS23001")
  
  runs = []
  if sim.args.action == 'setup':
    indirs, rndays = packed_inputs_to_unpacked_inputs(sim.args.packed_dir)
    runs = [{'inputs_dir': indir, 'rnday': rnday, 'outputs_dir': indir+"/outputs", 'output_files':['fort.61.nc']} for indir, rnday in zip(indirs, rndays)]
    print(runs)

   
  sim.run(
    runs=runs,
    execs_dir=os.path.expandvars("$WORK/execs/owi_wrapping"),
    inputs_dir=os.path.expandvars("$STOCKYARD/ls6/simulations/global-ml/global_mesh"),
    processors_per_node=50,
    no_writers=True,
    runtime=2/(len(runs)+1),
    node_count=10,
    maxJobNodes=10,
    queue="development"
  )

