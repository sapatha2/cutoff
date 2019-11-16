import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pyqmc.reblock import _reblock, reblock_summary, opt_block, optimally_reblocked
from numpy.polynomial import polynomial 

def analyze_hdf5(hdf_file):
    with h5py.File(hdf_file, 'r') as hdf:
        e = np.array([np.array(x) for x in list(hdf['pgradtotal'])])
        dppsi = np.array([np.array(x) for x in list(hdf['pgraddppsi'])])
        dpH = np.array([np.array(x) for x in list(hdf['pgraddpH'])])
    
    df = pd.DataFrame({'e':e, 'dppsi': dppsi[:,0]})
    for i in range(32):
        df['dpH_'+str(i)] = dpH[:,0,i]
    df.to_pickle('dedp.pickle')
    return e, dppsi, dpH

if __name__ == '__main__':
    #e, dppsi, dpH = analyze_hdf5('dedp_vmc.hdf5')
    #exit(0)
    df = pd.read_pickle('dedp.pickle')
    
    #Shuffle 
    print(np.isnan(df['dpH_0']).sum(), "isnan")
    df = df[np.isfinite(df['dpH_0'])]
    e = df['e'].values
    dppsi = df['dppsi'].values
    dpH = []
    print("Steps complete ", e.shape)
    for i in range(32):
      dpH.append(df['dpH_'+str(i)])
    dpH = np.array(dpH).T

    #Check timetraces for warmups
    '''
    plt.plot(dpH)
    plt.show()
    exit(0)
    '''

    #Plot histograms
    warmup = 20000
    e = e[warmup:]
    dppsi = dppsi[warmup:]
    dpH = dpH[warmup:]
    '''
    plt.hist(dpH[:, 0])
    plt.show()
    exit(0)
    '''

    #Reblock each quantity
    '''
    df = pd.DataFrame({'e':dpH[:,0]})
    nblocks = []
    err = []
    err_err = []
    nblocks = np.array([4000, 1000, 500, 200, 100, 80, 50, 40, 20, 10, 5])
    for nblock in nblocks:
        d = reblock_summary(df, nblock)
        err.append(d["standard error"])
        err_err.append(d["standard error error"])
    plt.errorbar(e.shape[0]/nblocks, err, err_err,fmt='o')
    plt.xlabel('step per block')
    plt.ylabel('error')
    plt.show()
    exit(0)
    '''

    #Evaluate errors after reblocking
    steps_per_block = 8000
    reblocked_e = np.array(_reblock(e, e.shape[0]/steps_per_block))
    reblocked_dppsi = np.array(_reblock(dppsi, dppsi.shape[0]/steps_per_block))
    reblocked_dpH = np.array([np.array(_reblock(dpH[:,i], dpH.shape[0]/steps_per_block))
                      for i in range(dpH.shape[1])]).T
    
    print("Reblocking")
    for v in [reblocked_e, reblocked_dppsi, reblocked_dpH[:, 0]]:
        print(v.mean(), v.std()/np.sqrt(v.shape[0]))
    print("Blocks remaining: ", v.shape[0])

    #Bootstrapping
    Nbs = 100

    print("Bootstrapping")
    for v in [reblocked_e, reblocked_dppsi, reblocked_dpH[:, 0]]:
        bs = []
        for i in range(Nbs):
            sample = np.random.choice(v.shape[0], v.shape[0], replace=True)
            sampled_v = v[sample]
            bs.append(sampled_v.mean())
        bs = np.array(bs)
        print(bs.mean(), bs.std())

    #dEdp Bootstrap 
    dEdp = []
    for i in range(Nbs):
        sample = np.random.choice(v.shape[0], v.shape[0], replace=True)
        sampled_e = reblocked_e[sample]
        sampled_dppsi = reblocked_dppsi[sample]
        sampled_dpH = reblocked_dpH[sample]

        bs_e = sampled_e.mean()
        bs_dppsi = sampled_dppsi.mean()
        bs_dpH = sampled_dpH.mean(axis=0)

        dEdp.append(bs_dpH- np.array([bs_e * bs_dppsi] * bs_dpH.shape[0]))
    dEdp = np.array(dEdp)

    means = dEdp.mean(axis=0)/1e-3 #mHa
    stds  = dEdp.std(axis=0)/1e-3
    cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.20])
    cutoffs = np.sort(cutoffs)
    
    #Fit below 1e-2 only
    means = means[cutoffs <= 0.12]
    stds =  stds[cutoffs <= 0.12]
    cutoffs = cutoffs[cutoffs <= 0.12]
    
    #Fit with upper bound for cubic scaling
    bound = 0.12
    fig, ax = plt.subplots(1, 1, figsize=(3,3))
    ax.errorbar(cutoffs, means, stds/4, fmt='o',color='tab:blue',zorder=1)
    #ax.errorbar(cutoffs, means, stds, fmt='o',color='tab:blue',zorder=0, alpha=0.5)

    p = polynomial.polyfit(cutoffs, means, [0,3])
    print("Polyfit : ", p)
    x = np.linspace(0, max(cutoffs), 100)
    ax.plot(x, p[0] + p[3] * x**3, '--',c='tab:orange', zorder=100)

    ax.plot(0, p[0], '*', c='tab:red',markersize=10,zorder=200)
    #p = polynomial.polyfit(cutoffs, means, [0,3,4])
    #print("Polyfit quartic: ", p)
    #ax.plot(x, p[0] + p[3] * x**3 + p[4]* x**4, '--',c='tab:orange', alpha=0.5,zorder=99)

    ax.set_xlabel(r'$\epsilon$ (Bohr)')
    ax.set_ylabel(r'$\partial E/\partial p$ (mHa)')
    plt.savefig('dedp.pdf',bbox_inches='tight')
