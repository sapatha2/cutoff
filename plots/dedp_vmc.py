import numpy as np 
import pyqmc
from pyscf import gto, scf, mcscf
from pyqmc.dasktools import distvmc
from dask.distributed import Client, LocalCluster
from pyqmc.accumulators import EnergyAccumulator, LinearTransform
from pyqmc_regr import PGradTransform_new
from wavefunction import wavefunction

if __name__ == '__main__':
  nconfig_per_core = 100
  ncore = 2
  nsteps = 100000

  cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20])
  cutoffs = np.sort(cutoffs)
  mol, mf, mc, wf, to_opt, freeze = wavefunction(return_mf=True)
  wf.parameters['wf1det_coeff'] *= 0 
  wf.parameters['wf1det_coeff'][[0,10]] = 1./np.sqrt(2.)

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
      nsteps_per=100,
      nsteps=nsteps,
      hdf_file='dedp_vmc_local.hdf5'
  )
