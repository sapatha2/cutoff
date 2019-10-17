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
ynode = 1.5
distances = []
for i in range(df.shape[0]):
  coords = df.iloc[i]['configs']
  dfromnode = np.abs(coords[0,0,1] - 1.5)
  distances.append(dfromnode)
ind = np.argsort(distance)
exit(0)

mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf,ncas=4,nelecas=(2,0))
mc.kernel()
wf, to_opt, freeze = pyqmc.default_multislater(mol, mf, mc) 

#Initial run
