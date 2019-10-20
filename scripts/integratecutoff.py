import pyqmc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.integrate as integrate
from pyscf import gto, scf, mcscf
from pyqmc_regr import PGradTransform
from pyqmc import MultiSlater 
from pyqmc.coord import OpenConfigs
from pyqmc.accumulators import EnergyAccumulator, LinearTransform 

def dpH(x, pgrad, pgrad_bare, node_coords, gradient):
  """Bias of dpH across a node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)
  d = pgrad(coords, wf)
  dpH = np.array(d['dpH'])[:, 0]
   
  d_bare = pgrad_bare(coords, wf)
  dpH_bare = np.array(d_bare['dpH'])[:, 0]

  return (dpH - dpH_bare) * np.exp(2 * val[1])

def dpH2(x, pgrad, node_coords, gradient):
  """Variance dpH^2 across a node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)  
  d = pgrad(coords, wf)
  dpH2 = np.array(d['dpH'])[:, 0]**2 * np.exp(2 * val[1])
  return dpH2

def norm(x, node_coords, gradient):
  """integral normalization across a node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)  
  return np.exp(2 * val[1])

if __name__ == '__main__':
  mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
  mf = scf.RHF(mol).run()
  mc = mcscf.CASCI(mf,ncas=4,nelecas=(2,0))
  mc.kernel()
  wf, to_opt, freeze = pyqmc.default_multislater(mol, mf, mc) 

  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  pgrad_bare = PGradTransform(eacc, transform, nodal_cutoff = 1e-10)

  node_coords = np.load('coords.pickle')
  gradient = np.load('grad.pickle')
  biases = []
  variances = []
  normalizations = []
  cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075])
  for cutoff in cutoffs:
    pgrad = PGradTransform(eacc, transform, nodal_cutoff = cutoff)
    bias = integrate.quad(lambda x: dpH(x, pgrad, pgrad_bare, node_coords, gradient), -cutoff, cutoff, epsabs = 1e-11, epsrel = 1e-11)
    variance = integrate.quad(lambda x: dpH2(x, pgrad, node_coords, gradient),  -cutoff, cutoff, epsabs = 1e-11, epsrel = 1e-11)
    normalization = integrate.quad(lambda x: norm(x, node_coords, gradient), -cutoff, cutoff, epsabs = 1e-11, epsrel = 1e-11)
    biases.append(bias[0])
    variances.append(variance[0])
    normalizations.append(normalization[0])
  
  df = pd.DataFrame({'cutoff': cutoffs, 'bias': biases, 'variance': variances, 'normalization': normalizations})
  df.to_pickle('integratecutoff.pickle')
