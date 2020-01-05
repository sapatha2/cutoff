import pyqmc
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
from pyscf import gto, scf, mcscf
from pyqmc_regr import PGradTransform_new
from pyqmc import MultiSlater 
from pyqmc.coord import OpenConfigs
from pyqmc.accumulators import EnergyAccumulator, LinearTransform, PGradTransform

def psi2(x, node_coords, gradient, wf):
  """Wavefunction squared across the node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)
  return np.exp(val[1] * 2)

def dpH(x, pgrad, pgrad_bare, node_coords, gradient, wf):
  """Bias of dpH across a node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)
  d = pgrad(coords, wf)
  dpH = np.array(d['dpH'])[:, 0, 0]
   
  d_bare = pgrad_bare(coords, wf)
  dpH_bare = np.array(d_bare['dpH'])[:, 0]

  return (dpH - dpH_bare) * np.exp(2 * val[1])

def dpH2(x, pgrad, node_coords, gradient, wf):
  """Variance dpH^2 across a node"""
  coords = OpenConfigs(gradient * x + node_coords)
  val = wf.recompute(coords)  
  d = pgrad(coords, wf)
  dpH2 = np.array(d['dpH'])[:, 0, 0]**2 * np.exp(2 * val[1])
  return dpH2

def viznode(node_coords, node_grad, cutoffs, vizfile='viznode.pdf'):
  from wavefunction import wavefunction 
  mol, wf, to_opt, freeze = wavefunction()
  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  
  #Generate the distances we want 
  x = np.arange(np.sqrt(1/0.01), 1001)
  x = 1./x**2
  x = np.append(x, np.linspace(0.01, 0.12, 100))
  x = np.append(x, np.array([0, 1e-15, 1e-12, 1e-8] + [-y for y in x]))
  x = np.sort(x)
  
  coord_path = np.einsum('ij,l->lij', node_grad[0], x) + node_coords[0]
  coord_path = OpenConfigs(coord_path)

  #Move across the node in this path
  val = wf.recompute(coord_path)
  fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (3,6), sharex=True)
  for k, cutoff in enumerate([1e-8]+cutoffs):
    pgrad = PGradTransform_new(eacc, transform, nodal_cutoff = np.array([cutoff]))
    d = pgrad(coord_path, wf)

    total = d['total']
    dpH = np.array(d['dpH'])[:, 0, 0]

    if(cutoff == 1e-8):
      ax[0].plot(x, dpH *  (val[0]*np.exp(val[1]))**2, 'k-', label = r'$10^{'+str(int(np.log10(cutoff)))+'}$')
      ax[1].plot(x, np.log10(dpH**2 * (val[0]*np.exp(val[1]))**2), 'k-')
    else:
      ax[0].plot(x, dpH *  (val[0]*np.exp(val[1]))**2, '-', label = r'$10^{'+str(int(np.log10(cutoff)))+'}$')
      ax[1].plot(x, np.log10(dpH**2 * (val[0]*np.exp(val[1]))**2), '-')

  ax[0].set_ylabel(r'$E_L\frac{\partial_p \Psi}{\Psi} f_\epsilon |\Psi|^2$')
  ax[1].set_ylabel(r'log$_{10}((E_L\frac{\partial_p \Psi}{\Psi})^2 f_\epsilon^2|\Psi|^2)$')
  ax[1].set_xlabel(r'$x$ (Bohr)')
  ax[0].set_xlim((-max(x) - 0.02,max(x) + 0.02))
  ax[1].set_xlim((-max(x) - 0.02,max(x) + 0.02))
  ax[0].legend(loc='best',title=r'$\epsilon$ (Bohr)')
  plt.savefig(vizfile ,bbox_inches='tight')
  plt.close()

def integratenode(node_coords, node_grad, vizfile='integratenode.pdf', integ_range = 1e-1, poly = 1e-2, max_cutoff=0.1):
  from wavefunction import wavefunction 
  import scipy.integrate as integrate
  from numpy.polynomial import polynomial
  mol, wf, to_opt, freeze = wavefunction()
  
  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  pgrad_bare = PGradTransform(eacc, transform, 1e-15)
  #Integrate biases and variances
  
  biases = []
  biases_err = []
  variances = []
  cutoffs = list(np.logspace(-6, -1, 20)) + [0.05, 0.075]
  '''
  normalization = integrate.quad(lambda x: psi2(x, node_coords, node_grad, wf), -integ_range, integ_range, epsabs = 1e-15, epsrel = 1e-15)
  for cutoff in cutoffs:
    print(cutoff)
    pgrad = PGradTransform_new(eacc, transform, nodal_cutoff = np.array([cutoff]))
    bias = integrate.quad(lambda x: dpH(x, pgrad, pgrad_bare, node_coords, node_grad, wf)/1e-40, -cutoff, cutoff, epsabs = 1e-15, epsrel = 1e-15, points=[0])
    variance = integrate.quad(lambda x: dpH2(x, pgrad, node_coords, node_grad, wf),  -cutoff, cutoff, epsabs = 1e-15, epsrel = 1e-15, points=[0])
    variance += integrate.quad(lambda x: dpH2(x, pgrad, node_coords, node_grad, wf),  -integ_range + cutoff, integ_range - cutoff, epsabs = 1e-15, epsrel = 1e-15)
    biases.append(bias[0]*1e-40/normalization[0])
    variances.append(variance[0]/normalization[0])
  df = pd.DataFrame({'cutoff': cutoffs, 'bias': biases, 'variance': variances})
  df.to_pickle('integratenode.pickle')
  ''' 
  df = pd.read_pickle('integratenode.pickle')
  #Fit theory curves and visualize
  ind = np.argsort(df['cutoff'])
  
  ind = ind[df['cutoff'].iloc[ind]<=max_cutoff]

  fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (3,6), sharex=True)
  x = df['cutoff'].iloc[ind]
  y = np.abs(df['bias']).iloc[ind]
  y = y[2:]
  x = x[2:]
  
  p = polynomial.polyfit(x, y, [3])
  print("Fit for bias ", p)
  xfit = np.linspace(min(x), max(x[x<poly]), 1000)
  fit = p[3] * (xfit) ** 3
  ax[0].plot(np.log10(x), np.log10(y), 'o')
  ax[0].plot(np.log10(xfit), np.log10(fit), '--')
  ax[0].set_ylabel(r'log$_{10}$(Bias)')
 
  x = df['cutoff'].iloc[ind]
  y = df['variance'].iloc[ind]
  y = y[2:]
  x = x[2:]
  x = np.log10(x)
  y = np.log10(y)
  poly = np.log10(poly)
  p = polynomial.polyfit(x[x<poly], y[x<poly], [1,0])
  print("Fit for variance ", p)
  xfit = np.logspace(min(x),max(x[x<=poly]), 1000)
  fit = p[0] + p[1] * np.log10(xfit)
  ax[1].plot(x, y, 'o')
  ax[1].plot(np.log10(xfit), fit, '--')
  ax[1].set_xlabel(r'$log_{10}(\epsilon/$Bohr$)$')
  ax[1].set_ylabel(r'log$_{10}$(Variance)')
  #ax[1].set_xlim((-3.2, 2.3))
  #ax[1].set_xticks(np.arange(-3,3))
 
  plt.savefig(vizfile,bbox_inches='tight')
  plt.close()
