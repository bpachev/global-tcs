import sys
import netCDF4 as nc
import numpy as np

"""Script to compress maxele.63.nc
"""

if __name__ == "__main__":
    outputsdir = sys.argv[1]
    rundir = sys.argv[2]

    with nc.Dataset(rundir+"/maxele.63.nc") as ds:
        zeta = ds["zeta_max"][:]
        zeta_time = ds["time_of_zeta_max"][:]

    np.savez(outputsdir+"/maxele",
        zeta=zeta.astype(np.float32),
        zeta_time=zeta_time.astype(np.float32)
    )



