import numpy as np 
import pandas as pd
import h5py

def analyze_hdf5(hdf_file):
    with h5py.File(hdf_file, 'r') as hdf:
        print(list(hdf.keys()))
        dpH = np.array([np.array(x) for x in list(hdf['pgraddpH'])])
        dppsi = np.array([np.array(x) for x in list(hdf['pgraddppsi'])])
        e = np.array([np.array(x) for x in list(hdf['pgradtotal'])])
    
    dEdp = dpH - np.einsum('i,ij->ij',e,dppsi)[:, np.newaxis] #(nstep, nparm, cutoff) #Raw

    dEdp_mu = np.mean(dEdp, axis=0)[0] #Only a single parameter evaluated
    dEdp_std = np.std(dEdp, axis=0)[0]/np.sqrt(dEdp.shape[0])
   
    cutoffs = list(np.logspace(-8, -1, 20)) + list([0.05,0.075])
    df = pd.DataFrame({'cutoff':cutoffs, 'dEdp': dEdp_mu, 'err': dEdp_std})
    df.to_json('dedp_vmc.json')

if __name__ == '__main__':
    analyze_hdf5('dedp_vmc.hdf5')
