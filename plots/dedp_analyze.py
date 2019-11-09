def analyze_hdf5(hdf_file):
    import h5py
    with h5py.File(hdf_file, 'r') as hdf:
        print(list(hdf['pgraddpH']))
        print(list(hdf['pgradnode_cut']))
if __name__ == '__main__':
    analyze_hdf5('dedp_vmc.hdf5')
