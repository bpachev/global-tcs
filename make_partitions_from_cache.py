import numpy as np
import h5py

# script to map mesh nodes to processors
# essentially reads fort.80

fname = "/scratch1/08009/bpachev/ch-sim/jobs/tc_ensemble_20250207_121935/cache-dir/fort.80"

def read_fort80(fort80, outfile="node_to_process.hdf5"):
  """Read the process-to-node mapping from an ADCIRC fort.80 file and convert to a Numpy format.
  """

  LINES_BEFORE_NUM_PE = 4
  LINES_AFTER_NUM_PE = 9
  NODES_PER_LINE = 9

  node_lists = []
  max_node = 0
  with open(fort80, 'r') as fp:
    for i in range(LINES_BEFORE_NUM_PE): fp.readline()
    num_procs = int(fp.readline().strip().split("!")[0])
    print("Num processors", num_procs)

    for i in range(LINES_AFTER_NUM_PE): fp.readline()

    for i in range(num_procs):
      line = fp.readline()
      if "!" not in line:
        raise RuntimeError(f"Expected comment in line '{line.strip()}'!")
      pe_num, num_nodes = map(int, line.split()[:2])
      expected_lines = (num_nodes + NODES_PER_LINE-1) // NODES_PER_LINE
      nodes = []
      for j in range(expected_lines):
        nodes.extend(map(int, fp.readline().split()))
      max_node = max(max_node, max(nodes))
      node_lists.append(nodes)

  node_to_proc = np.zeros(max_node, dtype=int)
  for proc_num, nodes in enumerate(node_lists):
    node_to_proc[np.array(nodes)-1] = proc_num
  
  counts = np.bincount(node_to_proc)
  print("Nodes per process", counts)
  assert counts[0] <= len(node_lists[0])
  
  with h5py.File(outfile, "w") as outds:
    outds["node_to_proc"] = node_to_proc

if __name__ == "__main__":
  read_fort80(fname)
