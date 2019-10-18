import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import pyqmc
from pyqmc.coord import OpenConfigs
from pyqmc import MultiSlater 
from pyscf import gto, scf, mcscf
import copy 

df = pd.read_pickle('findanode.pickle')
'''
#Locate a node in the wave function 
plt.plot(df['ycoord'], df['total'], 'o')
plt.show()

plt.plot(df['ycoord'], df['dpH'], 'o')
plt.show()

plt.plot(df['ycoord'], df['dppsi'], 'o')
plt.show()
'''

#Found a node at ycoord = 1.5
#Let's find the direction perpendicular to it
distances = []
ind = np.argsort(-df['dpH'].values)
node_coords = OpenConfigs(df.iloc[ind[0]]['configs'])

mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf,ncas=4,nelecas=(2,0))
mc.kernel()
wf, to_opt, freeze = pyqmc.default_multislater(mol, mf, mc) 
val = wf.recompute(node_coords)

#Get the gradient
nelec = mol.nelec[0] + mol.nelec[1]
grad_n = []
for e in range(nelec):
  node_grad = wf.gradient(e, node_coords.electron(e))
  node_grad /= (np.linalg.norm(node_grad) * np.sqrt(nelec))
  grad_n.append(node_grad)
grad_n = np.array(grad_n)

#Generate the distances we want 
x = list(np.logspace(0, 0.5, 100) - 1 + 1e-6)
x += [-y for y in x]
x = np.sort(x)
coord_path = np.einsum('ij,l->lij',grad_n[:, :, 0], x) + node_coords.configs[np.newaxis, 0, :, :]
df = {'x':list(x)*12, 'coord_path': coord_path.reshape(200*4*3).tolist()}
df = pd.DataFrame(df)
df.to_pickle('sampleperp.pickle')
