import numpy as np 
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from pyqmc.reblock import _reblock, reblock_summary, opt_block, optimally_reblocked

def analyze_hdf5(hdf_file):
    with h5py.File(hdf_file, 'r') as hdf:
        e = np.array([np.array(x) for x in list(hdf['pgradtotal'])])
        dppsi = np.array([np.array(x) for x in list(hdf['pgraddppsi'])])
        dpH = np.array([np.array(x) for x in list(hdf['pgraddpH'])])
    return e, dppsi, dpH

if __name__ == '__main__':
    e, dppsi, dpH = analyze_hdf5('dedp_vmc.hdf5')
    
    #Check timetraces for warmups
    dppsi = dppsi[:, 0]
    dpH = dpH[:, 0]
    '''
    plt.plot(dpH[:, :25])
    plt.show()
    exit(0)
    '''

    #Plot histograms
    warmup = 400
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
    steps_per_block = 50
    reblocked_e = np.array(_reblock(e, e.shape[0]/steps_per_block))
    reblocked_dppsi = np.array(_reblock(dppsi, dppsi.shape[0]/steps_per_block))
    reblocked_dpH = np.array([np.array(_reblock(dpH[:,i], dpH.shape[0]/steps_per_block))
                      for i in range(dpH.shape[1])]).T
    
    print("Reblocking")
    for v in [reblocked_e, reblocked_dppsi, reblocked_dpH[:, 0]]:
        print(v.mean(), v.std()/np.sqrt(v.shape[0]))

    #Bootstrapping
    Nbs = 400

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

    means = dEdp.mean(axis=0)
    stds  = dEdp.std(axis=0)
    cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    cutoffs = np.sort(cutoffs)
    plt.errorbar(cutoffs, means, stds, fmt='o')
    plt.show()
