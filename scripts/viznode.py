import pyqmc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from pyscf import gto, scf, mcscf
from pyqmc_regr import PGradTransform
from pyqmc import MultiSlater 
from pyqmc.coord import OpenConfigs
from pyqmc.accumulators import EnergyAccumulator, LinearTransform 

def dpH(x, pgrad, pgrad_bare, node_coords, gradient, wf):
  """Bias of dpH across a node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)
  d = pgrad(coords, wf)
  dpH = np.array(d['dpH'])[:, 0]
   
  d_bare = pgrad_bare(coords, wf)
  dpH_bare = np.array(d_bare['dpH'])[:, 0]

  return (dpH - dpH_bare) * np.exp(2 * val[1])

def dpH2(x, pgrad, node_coords, gradient, wf):
  """Variance dpH^2 across a node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)  
  d = pgrad(coords, wf)
  dpH2 = np.array(d['dpH'])[:, 0]**2 * np.exp(2 * val[1])
  return dpH2

def viznode(node_coords, node_grad, cutoffs, vizfile='viznode.pdf'):
  from wavefunction import wavefunction 
  mol, wf, to_opt, freeze = wavefunction()

  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  
  #Generate the distances we want 
  x = np.arange(np.sqrt(2), 101)
  x = 1./x**2
  x = np.append(x, np.array([0] + [-y for y in x]))
  x = np.sort(x)
  coord_path = np.einsum('ij,l->lij', node_grad[0], x) + node_coords[0]
  coord_path = OpenConfigs(coord_path)

  #Move across the node in this path
  val = wf.recompute(coord_path)
  fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (3,6), sharex=True)
  for k, cutoff in enumerate(cutoffs):
    pgrad = PGradTransform(eacc, transform, nodal_cutoff = cutoff)
    d = pgrad(coord_path, wf)

    total = d['total']
    dpH = np.array(d['dpH'])[:, 0]

    ax[0].plot(x, np.sign(dpH) * np.log10(abs(dpH)), '-', label = str(cutoff))
    ax[1].plot(x, np.log10(dpH**2 * (val[0]*np.exp(val[1]))**2), '-')

  ax[0].set_ylabel(r'sgn$(\frac{H\Psi}{\Psi} \frac{\partial_p\Psi}{\Psi})$ x log$_{10}|\frac{H\Psi}{\Psi} \frac{\partial_p\Psi}{\Psi}|$')
  ax[1].set_ylabel(r'log$_{10}(|\frac{H\Psi}{\Psi} \frac{\partial_p\Psi}{\Psi}|^2 |\Psi|^2)$')
  ax[1].set_xlabel('x (Bohr)')
  ax[0].set_xlim((-max(cutoffs),max(cutoffs)))
  ax[1].set_xlim((-max(cutoffs),max(cutoffs)))
  ax[0].legend(loc='best',title=r'$\epsilon$')
  plt.savefig(vizfile ,bbox_inches='tight')
  plt.close()

def integratenode(node_coords, node_grad, cutoffs, vizfile='integratenode.pdf', scalebias = 1e-12, poly = 1e-2):
  from wavefunction import wavefunction 
  import scipy.integrate as integrate
  from numpy.polynomial import polynomial
  mol, wf, to_opt, freeze = wavefunction()

  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  pgrad_bare = PGradTransform(eacc, transform, nodal_cutoff = 1e-10)

  #Integrate biases and variances
  biases = []
  variances = []
  cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075])
  for cutoff in cutoffs:
    pgrad = PGradTransform(eacc, transform, nodal_cutoff = cutoff)
    bias = integrate.quad(lambda x: dpH(x, pgrad, pgrad_bare, node_coords, node_grad, wf), -cutoff, cutoff, epsabs = 1e-15, epsrel = 1e-15)
    variance = integrate.quad(lambda x: dpH2(x, pgrad, node_coords, node_grad, wf),  -cutoff, cutoff, epsabs = 1e-15, epsrel = 1e-15)
    biases.append(bias[0])
    variances.append(variance[0])
  df = pd.DataFrame({'cutoff': cutoffs, 'bias': biases, 'variance': variances})

  #Fit theory curves and visualize
  ind = np.argsort(df['cutoff'])

  fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (3,6), sharex=True)
  x = df['cutoff'].iloc[ind]
  y = df['bias'].iloc[ind]/scalebias
  p = polynomial.polyfit(x[x>poly], y[x>poly], [3,0])
  xfit = np.linspace(min(x[x>poly]), max(x), 1000)
  fit = p[0] + p[3] * xfit ** 3
  ax[0].plot(np.log10(x), y, 'o')
  ax[0].plot(np.log10(xfit), fit, '--')
  ax[0].set_ylabel(r'Bias/$10^{-12}$')

  x = np.log10(df['cutoff'].iloc[ind])
  y = np.log10(df['variance'].iloc[ind])
  poly = np.log10(poly)
  p = polynomial.polyfit(x[x<=poly], y[x<=poly], 1)
  xfit = np.logspace(min(x),max(x[x<=poly]), 1000)
  fit = p[0] + p[1] * np.log10(xfit)
  ax[1].plot(x, y, 'o')
  ax[1].plot(np.log10(xfit), fit, '--')
  ax[1].set_xlabel(r'log$_{10}(\epsilon)$')
  ax[1].set_ylabel(r'log$_{10}$(Variance)')
  plt.savefig(vizfile,bbox_inches='tight')
  plt.close()
