import pyqmc
import pandas as pd
import numpy as np 
from pyqmc.coord import OpenConfigs
from wavefunction import wavefunction
from viznode import viznode, integratenode

def sweepelectron():
  """
  Sweep an electron across the molecule
  to find a guess for the nodal position
  """
  import copy 
  from pyqmc.accumulators import EnergyAccumulator, LinearTransform, PGradTransform
  
  #Generate wave function and bare parameter gradient objects
  mol, wf, to_opt, freeze = wavefunction() 
  eacc = EnergyAccumulator(mol)
  transform = LinearTransform(wf.parameters, to_opt, freeze)
  pgrad = PGradTransform(eacc, transform, 0)

  #Initial coords
  configs = pyqmc.initial_guess(mol, 1).configs[:,:,:]

  #Sweep electron 0
  full_df = None
  e = 0 #electron
  dim = 1 #Coordinate to vary
  
  for i in np.linspace(-5, 5, 200):
    new_configs = copy.deepcopy(configs)
    new_configs[:,e,dim] += i
    shifted_configs = OpenConfigs(new_configs)
    wfval = wf.recompute(shifted_configs)
    d = pgrad(shifted_configs, wf)

    small_df = pd.DataFrame({
      'ke':[d['ke'][0]],
      'total':[d['total'][0]],
      'dppsi':[d['dppsi'][0][-1]],
      'dpH'  :[d['dpH'][0][-1]],
      'wfval':[wfval[0][0]*np.exp(wfval[1][0])],
      'ycoord': i,
      'configs':[copy.deepcopy(new_configs)],
    })
    if(full_df is None): full_df = small_df
    else: full_df = pd.concat((full_df, small_df), axis=0)
  
  return full_df.reset_index()

def vizsweep(df):
  """
  Visualize your sweep to make sure it's decent
  """
  
  import matplotlib.pyplot as plt 
  ind = np.argsort(-df['dpH'].values)
  
  plt.plot(df['ycoord'], df['wfval'], 'o')
  plt.plot(df['ycoord'].iloc[ind[0]], df['wfval'].iloc[ind[0]], 'o')
  plt.show()

  plt.plot(df['ycoord'], df['dpH'], 'o')
  plt.plot(df['ycoord'].iloc[ind[0]], df['dpH'].iloc[ind[0]], 'o')
  plt.show()

def locatenode(df, scale):
  """
  Pinpoint a node, scale variable scales wf value
  so that minimization algorithm can converge
  """

  from scipy.optimize import minimize_scalar
  mol, wf, to_opt, freeze = wavefunction() 
  
  ind = np.argsort(-df['dpH'].values)
  node_coords_0 = np.array(df.iloc[ind[0]]['configs']) #Pretty close!

  #Get the gradient
  def wfgrad(coords, wf, mol):
    nelec = mol.nelec[0] + mol.nelec[1]
    val = wf.recompute(coords)
    grad = []
    for e in range(nelec):
      node_grad = wf.gradient(e, coords.electron(e)) * np.exp(val[1]) * val[0]
      grad.append(node_grad)
    grad = np.array(grad)
    grad /= np.linalg.norm(grad.ravel())
    return np.rollaxis(grad, -1) 

  #Value function along that gradient
  def wvfal(x, wf, gradient):
      node_coords = OpenConfigs(node_coords_0 + gradient * x)
      val = wf.recompute(node_coords)
      return np.exp(2 * val[1])/scale #Scaling for minimization 

  #Minimize function 
  node_coords = node_coords_0

  for i in range(1):
    val = wf.recompute(OpenConfigs(node_coords))
    grad = wfgrad(OpenConfigs(node_coords), wf, mol)
    print("Wfval: ", np.exp(val[1])*val[0])

    res = minimize_scalar(lambda x: wvfal(x, wf, grad), bracket = [-0.1, 0.1], tol = 1e-16)
    print("x: ",res.x)

    #Upgrade gradient
    node_coords += grad * res.x[0]

  return node_coords, grad

if __name__ == '__main__':
  """First sweep to make sure you have a potential node""" 
  #Run these first, once you hit a node DON'T run them again!
  #sweepdf = sweepelectron()
  #sweepdf.to_json('sweepelectron.json')
  #vizsweep(sweepdf)

  """Pinpoint the location of the node"""
  sweepdf = pd.read_json('sweepelectron.json')
  node_coords, node_grad = locatenode(sweepdf, scale = 1e-50)

  """Visualize the node"""
  cutoffs = [1e-5, 1e-3, 1e-2, 1e-1]
  viznode(node_coords, node_grad, cutoffs)

  """Integrate across the node"""
  integratenode(node_coords, node_grad, poly=1e-2, integ_range = 0.1) 
