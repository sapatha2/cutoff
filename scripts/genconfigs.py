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
from pyqmc.accumulators import EnergyAccumulator, LinearTransform 
from pyqmc_regr import PGradTransform 
from pyqmc.slateruhf import PySCFSlaterUHF

#Generate a ton of configurations!
mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 1.5", basis="cc-pvtz", unit="bohr", spin=0)
mf = scf.ROHF(mol).run()
mc = mcscf.CASCI(mf,ncas=4,nelecas=(1,1))
mc.kernel()
wf, to_opt, freeze = pyqmc.default_multislater(mol, mf, mc) 

parm = -1
mf.mo_coeff = mc.mo_coeff
mf.mo_occ *= 0 
mf.mo_occ[wf._det_occup[0][parm]] = 2
wfp = PySCFSlaterUHF(mol, mf)

coords = pyqmc.initial_guess(mol, 100000)

eacc = EnergyAccumulator(mol)
transform = LinearTransform(wf.parameters, to_opt, freeze)
pgrad_bare = PGradTransform(eacc, transform, nodal_cutoff = 1e-8)

warmup = 10
for i in range(100 + warmup):
  df, coords = vmc(wfp, coords, nsteps = 1)
  
  print(i)
  if(i > warmup):
    coords.configs.dump('vmc/coords'+str(i-warmup)+'.pickle')
    
    val = wf.recompute(coords)
    valp = wfp.value()
    
    d = pgrad_bare(coords, wf)

    data = {'dpH': np.array(d['dpH'])[:,parm], 'dppsi': np.array(d['dppsi'])[:,parm], 
        'en': np.array(d['total']),"wfval": val[1], "wfpval": valp[1]} 
    pd.DataFrame(data).to_json('vmc/evals'+str(i-warmup)+'.json')
