import pyqmc
import pandas as pd
import numpy as np 
from pyqmc.coord import OpenConfigs
from pyqmc import MultiSlater 
from pyscf import gto, scf, mcscf
from pyqmc.accumulators import EnergyAccumulator, LinearTransform, PGradTransform
import copy 

################
nconf = 1
cutoff = 0
################

mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.RHF(mol).run()
mc = mcscf.CASCI(mf,ncas=4,nelecas=(1,1))
mc.kernel()
wf, to_opt, freeze = pyqmc.default_multislater(mol, mf, mc)

eacc = EnergyAccumulator(mol)
transform = LinearTransform(wf.parameters, to_opt, freeze)
pgrad = PGradTransform(eacc, transform, nodal_cutoff = cutoff)

#Initial run
configs = pyqmc.initial_guess(mol, nconf).configs[:,:,:]

full_df = None
e = 0 #electron
dim = 1 #Coordinate to vary
configs[:,e,:] = [[0,0,0.75]] #Start at center of molecule
for i in np.linspace(-2, 2, 100):
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
  full_df.to_pickle('sweepelectron.pickle')
