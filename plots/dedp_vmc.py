import numpy as np 
import pyqmc
from pyscf import gto, scf, mcscf
from pyqmc.dasktools import distvmc
from dask.distributed import Client, LocalCluster
from pyqmc.accumulators import EnergyAccumulator, LinearTransform
from pyqmc_regr import PGradTransform_new
from wavefunction import wavefunction

if __name__ == '__main__':
  nconfig_per_core = 100 #5000
  ncore = 2
  cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075])
  cutoffs = np.sort(cutoffs)
  mol, mf, mc, wf, to_opt, freeze = wavefunction(return_mf=True)
  
  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  pgrad = PGradTransform_new(eacc, transform, np.array(cutoffs))

  #Client 
  cluster = LocalCluster(n_workers=ncore, threads_per_worker=1)
  client = Client(cluster)

  distvmc(
      wf,
      pyqmc.initial_guess(mol,nconfig_per_core * ncore),
      client=client,
      accumulators = {"pgrad": pgrad}, 
      nsteps_per=1000,
      nsteps=1000,
      hdf_file='dedp_vmc.hdf5'
  )