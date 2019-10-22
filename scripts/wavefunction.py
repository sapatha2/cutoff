from pyscf import gto, scf, mcscf
import numpy as np 
from scipy.optimize import curve_fit

def test_basis():
    """
    Basis set extrapolation using exponential 
    from "Basis-set convergence of the energy in molecular Hartree–Fock calculations"
    """
    
    import matplotlib.pyplot as plt
    import pandas as pd
    
    e = []
    ls = []

    bases = ["cc-pvdz", "cc-pvtz", "cc-pvqz", "cc-pv5z"]
    for l, basis in enumerate(bases):
      mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 3.015", basis=basis, unit="bohr", spin=0, symmetry=True)
      mf = scf.RHF(mol).run()
      mf.analyze()
      e.append(mf.e_tot)
      ls.append(l+2)

    def func(x, a, b, c):
        return c + a * np.exp(-b * x)

    popt, pcov = curve_fit(func, np.array(ls), np.array(e))
    plt.plot(ls, e, 'o')
    x = np.linspace(min(ls), max(ls), 100)
    plt.plot(x, func(x, *popt),'-')
    plt.show()

    df = {'basis': bases + ['cbs'], 'e': e +  [func(10**10, *popt)], 'ediff': list(np.array(e) - func(10**10, *popt)) + [0]}
    df = pd.DataFrame(df)
    print(df)
    df.to_json('basis_extrap.json')

def test_ci():
    """
    Full CI over 
    Li 1s, 2s, 2p
    Hi 1s
    """

    from pyscf2qwalk import print_qwalk
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 3.015", basis='cc-pvqz', unit="bohr", spin=0)
    mf = scf.ROHF(mol).run()
    mc = mcscf.CASCI(mf,ncas=6,nelecas=(2,2))
    mc.kernel()

def wavefunction():
    """
    Returns Full CI wave function 
    multiplied by a Jastrow with cusp conditions applied 
    """

    from pyqmc import default_msj
    mol = gto.M(atom="Li 0. 0. 0.; H 0. 0. 3.015", basis='cc-pvqz', unit="bohr", spin=0)
    mf = scf.ROHF(mol).run()
    mc = mcscf.CASCI(mf,ncas=6,nelecas=(2,2))
    mc.kernel()
   
    wf, to_opt, freeze = default_msj(mol, mf, mc)

    #Only need to take derivatives wrt determinant coefficients
    to_opt = ['wf1det_coeff'] 
    freeze = {'wf1det_coeff': freeze['wf1det_coeff']}

    #Only need to take derivatives wrt the highest energy det
    #, will have least overlap with nodes!
    freeze = {'wf1det_coeff': np.ones(freeze['wf1det_coeff'].shape).astype(bool)}
    freeze['wf1det_coeff'][-1] = False

    return wf, to_opt, freeze

if __name__=='__main__':       
    test_basis()
    test_ci()
