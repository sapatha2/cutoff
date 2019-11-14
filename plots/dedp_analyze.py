import numpy as np 
import pandas as pd
import h5py

def bootstrap(dpH, e, dppsi, N):
    dEdps = []
    for i in range(N):
        sample = np.random.choice(dpH.shape[0], dpH.shape[0], replace=True)
        dpH_sample = dpH[sample]
        e[sample] = e[sample]
        dppsi[sample] = dppsi[sample]
        
        dEdp = np.mean(dpH_sample, axis=0) -\
               np.mean(e, axis=0) * np.mean(dppsi, axis=0)[:, np.newaxis]
        dEdps.append(dEdp)
    dEdps = np.array(dEdps)
    dEdp_mu = np.mean(dEdps, axis=0)
    dEdp_std = np.std(dEdps, axis=0)
    return dEdp_mu, dEdp_std

def analyze_hdf5(hdf_file, nsplit, nbootstrap):
    with h5py.File(hdf_file, 'r') as hdf:
        dpH = np.array([np.array(x) for x in list(hdf['pgraddpH'])])
        dppsi = np.array([np.array(x) for x in list(hdf['pgraddppsi'])])
        e = np.array([np.array(x) for x in list(hdf['pgradtotal'])])
    
    print(dpH.shape) 
    dpH = np.array(np.split(dpH, nsplit)).mean(axis=0)
    dppsi = np.array(np.split(dppsi, nsplit)).mean(axis=0)
    e = np.array(np.split(e, nsplit)).mean(axis=0)

    dEdp_mu, dEdp_std = bootstrap(dpH, e, dppsi, nbootstrap)
   
    cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075])
    cutoffs = np.sort(cutoffs)
    df = pd.DataFrame({'cutoff':cutoffs, 'dEdp': dEdp_mu, 'err': dEdp_std})
    df.to_json('dedp_vmc.json')
    return df

if __name__ == '__main__':
    analyze_hdf5('dedp_vmc.hdf5', 1000, 100)
    '''
    import matplotlib.pyplot as plt 
    import statsmodels.api as sm
    from statsmodels.sandbox.regression.predstd import wls_prediction_std
    from numpy.polynomial import polynomial 

    df = pd.read_json('dedp_vmc.json')
    
    x = df['cutoff']
    y = df['dEdp']
    err = df['err']

    err = err[x>=1e-2]
    y = y[x>=1e-2]
    x = x[x>=1e-2]
    

    ols = sm.WLS(y, sm.add_constant(x[:, np.newaxis]**3),weight=1/err**3)
    ols = ols.fit()

    prstd, iv_l, iv_u = wls_prediction_std(ols)
    #Fit cubic to each 
    p_l = polynomial.polyfit(x, iv_l, [0,3])
    p_u = polynomial.polyfit(x, iv_u, [0,3])

    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    ax.errorbar(x, y, yerr=err,fmt='ko')
    x = np.linspace(0,max(x),100)
    ax.fill_between(x, p_l[0] + p_l[3]* x**3, p_u[0] + p_u[3]*x**3, color='gray', alpha=0.5)
    ax.errorbar(x[0], ols.params[0], yerr = (p_l[0] - p_u[0])/2, fmt='s', c='g')
    ax.set_ylabel(r'$\partial E/\partial p$ (Ha)')
    ax.set_xlabel(r'$\epsilon$ (Bohr)')
    plt.savefig('dedp.pdf',bbox_inches='tight')
    '''
