import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pyqmc
from pyqmc.coord import OpenConfigs
from pyqmc import MultiSlater 
from pyscf import gto, scf, mcscf
import copy 
from scipy.optimize import minimize 
from pyqmc.mc import vmc
from pyqmc.accumulators import EnergyAccumulator, LinearTransform, PGradTransform
from pyqmc_regr import PGradTransform_new
from pyqmc.slateruhf import PySCFSlaterUHF
from wavefunction import wavefunction

def genconfigs(n):
  """
  Generate configurations and weights corresponding
  to the highest determinant in MD expansion
  """

  import os 
  os.system('mkdir -p vmc/')

  mol, mf, mc, wf, to_opt, freeze = wavefunction(return_mf=True)
  #Sample from the wave function which we're taking pderiv relative to
  mf.mo_coeff = mc.mo_coeff
  mf.mo_occ *= 0 
  mf.mo_occ[wf.wf1._det_occup[0][-1]] = 2
  wfp = PySCFSlaterUHF(mol, mf)

  #Lots of configurations
  coords = pyqmc.initial_guess(mol, 100000)

  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  pgrad_bare = PGradTransform(eacc, transform, 0)

  #Lots of steps
  warmup = 10
  for i in range(n + warmup + 1):
    df, coords = vmc(wfp, coords, nsteps = 1)
    
    print(i)
    if(i > warmup):
      coords.configs.dump('vmc/coords'+str(i-warmup)+'.pickle')
      
      val = wf.recompute(coords)
      valp = wfp.value()
      
      d = pgrad_bare(coords, wf)

      data = {'dpH': np.array(d['dpH'])[:,-1], 'dppsi': np.array(d['dppsi'])[:,-1], 
          'en': np.array(d['total']), "wfval": val[1], "wfpval": valp[1]} 
      pd.DataFrame(data).to_json('vmc/evals'+str(i-warmup)+'.json')
  return -1

def collectconfigs(n, dump_file):
  """
  Collect all the configurations from genconfig
  into a single place
  """
  dpH_total = []
  weight_total = []
  logweight_total = []
  distance_squared = []

  mol, wf, to_opt, freeze = wavefunction()

  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  pgrad_bare = PGradTransform_new(eacc, transform, 1e-20)

  for i in range(1,n + 1):
    print(i)
    coords = OpenConfigs(pd.read_pickle('vmc/coords'+str(i)+'.pickle'))
    df = pd.read_json('vmc/evals'+str(i)+'.json')

    wf.recompute(coords)
    print("Recomputed")
    node_cut, r2 = pgrad_bare._node_cut(coords, wf)
    print("nodes cut")

    dpH = df['dpH'].values
    wfval = df['wfval'].values
    wfpval = df['wfpval'].values

    logweight = 2 * (wfval - wfpval)
    weight = np.exp(logweight)

    dpH_total += list(dpH)
    weight_total += list(weight)
    logweight_total += list(logweight)
    distance_squared += list(r2)

    df = pd.DataFrame({
      'dpH': dpH_total,
      'weight_total': weight_total,
      'logweight_total': logweight_total,
      'distance_squared': distance_squared
      })
    df.to_json(dump_file)
  return df 

def eval_configs(data, cutoffs):
  """
  Plot distribution of collected configurations
  """
  
  mol, wf, to_opt, freeze = wavefunction()
  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)

  print("data loaded")
  r2 = data['distance_squared']
  weight = data['weight_total']

  d = {}
  for cutoff in list(cutoffs): 
      node_cut = r2 < cutoff ** 2
      print(cutoff, node_cut.sum())
      
      c = 7./(cutoff ** 6)
      b = -15./(cutoff ** 4)
      a = 9./(cutoff ** 2)

      l2 = r2[node_cut]
      dpH = np.copy(data['dpH'])
      dpH[node_cut] *= a * l2 + b * l2**2 + c * l2**3

      hist, bin_edges = np.histogram(np.log10(np.abs(dpH)[np.abs(dpH)>0]), bins = 200, density = True, weights = weight[np.abs(dpH)>0])
      d['hist'+str(cutoff)] = list(np.log10(hist)) + [None]
      d['bins'+str(cutoff)] = bin_edges
  df = pd.DataFrame(d)
  df.to_json('histogram.json')

def plot_configs(df, cutoffs):
  #Raw histogram
  for cutoff in cutoffs:
    hist = df['hist'+str(cutoff)][:-1]
    bin_edges = df['bins'+str(cutoff)]
    plt.plot(list(bin_edges[:-1]) + list(bin_edges[1:]), list(hist) + list(hist), '.', label = str(cutoff))
  plt.plot(list(bin_edges[:-1]) + list(bin_edges[1:]), np.array(list(-bin_edges[:-1]) + list(-bin_edges[1:])) -4.2, 'k--')
  plt.legend(loc='best')
  plt.close()
  #plt.show()

  #KDE
  for cutoff in cutoffs:
    hist = df['hist'+str(cutoff)][:-1]
    bin_edges = df['bins'+str(cutoff)]
    x = np.array(list(bin_edges[:-1]) + list(bin_edges[1:]))
    x = 10.**x
    x /= cutoff ** -2
    x = np.log10(x)
    y = np.array(list(hist) + list(hist))
    y = 10.**y 
    y /= cutoff ** 3
    y = np.log10(y)
    plt.plot(x, y, '.', label = str(cutoff))
    plt.plot(x, -1.5*x - 5, 'k--')
  plt.legend(loc='best')
  plt.show()

if __name__ == '__main__':
  n = 200
  #Only needs to be run once!
  #genconfigs(n) 
  #df = collectconfigs(n,'vmc/collected.json')

  #Needs to be rerun for plotting
  #cutoffs = [1e-8, 1e-5, 1e-3, 1e-2, 1e-1]
  #data = pd.read_json('vmc/collected.json')
  #eval_configs(data, cutoffs)

  cutoffs = [1e-5, 1e-3, 1e-2, 1e-1]
  plot_configs(pd.read_json('histogram.json'), cutoffs)
