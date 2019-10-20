import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pyqmc
from pyqmc.coord import OpenConfigs
from pyqmc import MultiSlater 
from pyscf import gto, scf, mcscf
import copy 
from scipy.optimize import minimize 

df = pd.read_pickle('sweepelectron.pickle')
'''
#Locate a node in the wave function 
plt.plot(df['ycoord'], df['total'], 'o')
plt.show()

plt.plot(df['ycoord'], df['dpH'], 'o')
plt.show()

plt.plot(df['ycoord'], df['dppsi'], 'o')
plt.show()
'''

#Found a node at ycoord ~ 1.5
#Let's find the direction perpendicular to it!
distances = []
ind = np.argsort(-df['dpH'].values)
node_coords_0 = df.iloc[ind[0]]['configs'] #Pretty close!

#Now let's move towards the node as close as possible 
mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf,ncas=4,nelecas=(2,0))
mc.kernel()
wf, to_opt, freeze = pyqmc.default_multislater(mol, mf, mc) 

#Get the gradient
def wfgrad(coords, wf, mol):
  nelec = mol.nelec[0] + mol.nelec[1]
  grad = []
  for e in range(nelec):
    node_grad = wf.gradient(e, coords.electron(e))
    node_grad /= (np.linalg.norm(node_grad) * np.sqrt(nelec))
    grad.append(node_grad)
  grad = np.array(grad)
  return np.rollaxis(grad, -1)

#Value function along that gradient
def wvfal(x, wf, gradient):
    node_coords = OpenConfigs(node_coords_0 + gradient * x)
    val = wf.recompute(node_coords)
    return np.exp(2 * val[1])/1e-20 #Scaling for minimization 

#Minimize function 
val_0 = wf.recompute(OpenConfigs(node_coords_0))
grad_0 = wfgrad(OpenConfigs(node_coords_0), wf, mol)
res = minimize(lambda x: wvfal(x, wf, grad_0), 0)
print("x: ",res.x)

#Upgrade gradient
node_coords = node_coords_0 + grad_0 * res.x[0]
val = wf.recompute(OpenConfigs(node_coords))
grad = wfgrad(OpenConfigs(node_coords), wf, mol)
print("Pre: ",np.exp(2*val_0[1]), "Post: ",np.exp(2*val[1]))

#Update gradient
grad.dump('grad.pickle')
node_coords.dump('coords.pickle')
