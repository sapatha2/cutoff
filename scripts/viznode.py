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

  #Generate the distances we want 
  x = np.arange(np.sqrt(2), 101)
  x = 1./x**2
  x = np.append(x, np.array([0] + [-y for y in x]))
  x = np.sort(x)
  coord_path = np.einsum('ij,l->lij',gradient[0], x) + node_coords[0]
  coord_path = OpenConfigs(coord_path)

  val = wf.recompute(coord_path)
  cutoffs = [1e-8, 1e-5, 1e-3, 1e-2, 0.1]
  cutoff_labels = [r'$10^{-8}$',r'$10^{-5}$',r'$10^{-3}$',r'$10^{-2}$',r'$10^{-1}$']
  fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (3,6), sharex=True)
  for k, cutoff in enumerate(cutoffs):
    pgrad = PGradTransform(eacc, transform, nodal_cutoff = cutoff)
    d = pgrad(coord_path, wf)

    total = d['total']
    dpH = np.array(d['dpH'])[:, 0]

    ax[0].plot(x, np.sign(dpH) * np.log10(abs(dpH)), '-', label = cutoff_labels[k])
    ax[1].plot(x, np.log10(dpH**2 * (val[0]*np.exp(val[1]))**2), '-')

  ax[0].set_ylabel(r'sgn$(\frac{H\Psi}{\Psi} \frac{\partial_p\Psi}{\Psi})$ x log$_{10}|\frac{H\Psi}{\Psi} \frac{\partial_p\Psi}{\Psi}|$')
  ax[1].set_ylabel(r'log$_{10}(|\frac{H\Psi}{\Psi} \frac{\partial_p\Psi}{\Psi}|^2 |\Psi|^2)$')
  ax[1].set_xlabel('x (Bohr)')
  ax[0].set_xlim((-0.05,0.05))
  ax[1].set_xlim((-0.05,0.05))
  ax[0].legend(loc='best',title=r'$\epsilon$')
  plt.savefig('viznode.pdf',bbox_inches='tight')
