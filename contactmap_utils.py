import numpy as np
import h5py

def get_coors(filename, atom="CA", pdb=None, chain=None, verbose=False):
    with h5py.File(filename) as f:
        if pdb is None:
            pdbs = [k for k in f[atom].keys()]
            pdb = pdbs[0]
            if verbose:
                print("PDBs", pdbs)
        if chain is None:
            chains = [k for k in f[atom][pdb].keys()]
            chain = chains[0]
            if verbose:
                print("{} chains".format(pdb), chains)
        ds_name = "{}/{}/{}".format(atom, pdb, chain)
        if verbose:
            print(ds_name)
        return f[ds_name][()]
             
        
def distance_matrix(coors):
    n = coors.shape[0]
    coors_a = np.repeat(coors.reshape(n, 1, 3), n, axis=1)
    coors_b = np.repeat(coors.reshape(1, n, 3), n, axis=0)
    dist_mat = np.linalg.norm(coors_a-coors_b, axis=2)
    return dist_mat
    

def contact_matrix(ids):
    n = ids.shape[0]
    id_a = np.repeat(ids.reshape(n, 1), n, axis=1)
    id_b = np.repeat(ids.reshape(1, n), n, axis=0)
    contact_mat = (id_a-id_b) == 1
    return contact_mat