from ch_sim import BaseSimulator
import os

class BenchmarkSim(BaseSimulator):

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
            )


if __name__ == "__main__":
  
  sim = BenchmarkSim(allocation="DMS23001")
  
  sim.run(
    execs_dir=os.path.expandvars("$WORK/execs/owi_wrapping"),
    inputs_dir=os.path.expandvars("$STOCKYARD/ls6/simulations/global-ml/global_mesh"),
    processors_per_node=50,
    runtime=2,
    node_count=10,
    queue="development"
  )

