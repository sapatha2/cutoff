import json
from pyscf import gto,scf,mcscf,fci,lo,ci
from pyscf.scf import ROHF, UHF,ROKS,UKS
import numpy as np
import pandas as pd
import pyscf2qwalk

df=json.load(open("trail.json"))
spins={'ScO':1,'TiO':2,'VO':3,'CrO':4,'MnO':5,'FeO':4,'CuO':1}
re={'ScO':1.668,
    'TiO':1.623,
    'VO':1.591,
    'CrO':1.621,
    'MnO':1.648,
    'FeO':1.616,
    'CuO':1.725,
    }

ions=['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Cu']

nd={'Sc':(1,0),'Ti':(2,0),'V':(3,0),'Cr':(5,0),'Mn':(5,0),'Fe':(5,1),
     'Cu':(5,4) } 

basis = 'vtz'
el = 'Cu'
charge = 0

molname=el+'O'
mol=gto.Mole()

mol.ecp={}
mol.basis={}
for e in [el,'O']:
  mol.ecp[e]=gto.basis.parse_ecp(df[e]['ecp'])
  mol.basis[e]=gto.basis.parse(df[e][basis])
mol.charge=charge
mol.spin=spins[molname]
print('spin',molname,mol.spin)
mol.build(atom="%s 0. 0. 0.; O 0. 0. %g"%(el,re[molname]),verbose=4)
m=ROHF(mol)

m=scf.newton(m)
energy=m.kernel()
m.analyze()

ncas=13
ncore=4
nelec=(mol.nelec[0]-ncore,mol.nelec[1]-ncore)
      
myci=mcscf.CASCI(m,ncas,nelec,ncore=ncore)
myci.run()

ci = myci.ci
e  = myci.e_tot
mo = myci.mo_coeff
d = pd.DataFrame({'energy':[e],'spin':[spin],'ci':[ci],'mo':[mo]})
d.to_json('casci.json')
