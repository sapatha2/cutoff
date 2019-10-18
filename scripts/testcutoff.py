import pyqmc
from pyqmc.coord import OpenConfigs
from pyqmc import MultiSlater 
from pyscf import gto, scf, mcscf
import copy 
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
from pyqmc.accumulators import EnergyAccumulator, LinearTransform 
from pyqmc_regr import PGradTransform

df = pd.read_pickle('sampleperp.pickle')
coords = np.array(df['coord_path'].values).reshape(200,4,3)
x = np.array(df['x'])[:200]

coords = OpenConfigs(coords)
mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf,ncas=4,nelecas=(2,0))
mc.kernel()
wf, to_opt, freeze = pyqmc.default_multislater(mol, mf, mc) 

ne = mol.nelec[0] + mol.nelec[1]
val = wf.recompute(coords)
d2 = 0.0 
for e in range(ne):
    d2 += np.sum(wf.gradient(e, coords.electron(e)) ** 2, axis=0)
distance = 1./np.sqrt(d2)
plt.ylabel('Approx. distance')
plt.xlabel('Distance')
plt.plot(x, distance, 'o')
plt.savefig('distanceref.pdf',bbox_inches='tight')
plt.close()

##################################################################
#CUTOFF 
#cutoff = 1e-10
##################################################################
eacc = EnergyAccumulator(mol)
transform = LinearTransform(wf.parameters, to_opt, freeze)

fig, ax = plt.subplots(nrows = 2, ncols = 1, figsize = (3,6))
cutoffs = [1e-10, 0.05, 0.1, 0.5]
for cutoff in cutoffs:
  pgrad = PGradTransform(eacc, transform, nodal_cutoff = cutoff)
  d = pgrad(coords, wf)

  total = d['total']
  dpH = np.array(d['dpH'])[:, 0]

  if cutoff > 1e-10: 
    ax[0].plot(x, dpH, '-.', label = str(cutoff))
    ax[1].plot(x, dpH**2 * (val[0]*np.exp(val[1]))**2, '-.', label = str(cutoff))
  else: 
    ax[0].plot(x, dpH, '-o', label = str(cutoff))
    ax[1].plot(x, dpH**2 * (val[0]*np.exp(val[1]))**2, '-o', label = str(cutoff))

ax[0].set_ylabel('HPsi/Psi dpPsi/Psi')
ax[1].set_ylabel('(HPsi/Psi dpPsi/Psi)^2 Psi^2')
ax[1].set_xlabel('l')
ax[0].set_xlim((-0.5,0.5))
ax[1].set_xlim((-0.5,0.5))
ax[0].set_ylim((-50, 200))
ax[1].set_ylim((-0.2e-8,1.4e-8))
ax[0].legend(loc='best',title='cutoff')
plt.savefig('testcutoff.pdf',bbox_inches='tight')
