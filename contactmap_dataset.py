import numpy as np
import pandas as pd
import h5py
import json
import torch
from torch.utils.data import Dataset
from contactmap_utils import distance_matrix, contact_matrix


class ContactMapDataset(Dataset):
    
    def __init__(self, h5_path, atom="CA", random=False, tensor=True, unsqueeze=True,
                 min_size=64, max_size=1024, random_seed=2020):
        self.f = h5py.File(h5_path)
        self.random = random
        self.tensor = tensor
        self.unsqueeze = unsqueeze
        self.min_size = min_size
        self.max_size = max_size
        self.build(h5_path, atom)
        self.rs = np.random.RandomState(seed=random_seed)
        
    def build(self, h5_path, atom):
        self.data = list()
        for pdb in self.f[atom].keys():
            for chain in self.f[atom][pdb].keys():
                ds_name = "{}/{}/{}".format(atom, pdb, chain)
                n, _ = self.f[ds_name].shape
                if n >= self.min_size and n <= self.max_size:
                    self.data.append(ds_name)
        self.n = len(self.data)
                    
    def __len__(self):
        return self.n
    
    def __getitem__(self, idx):
        coors = self.f[self.data[idx]]
        if self.random:
            ids = self.rs.permutation(coors.shape[0])
        else:
            ids = np.arange(coors.shape[0])
        dist_mat = distance_matrix(np.take(coors, ids, axis=0))
        cont_mat = contact_matrix(ids)
        if self.tensor:
            dist_mat = torch.from_numpy(dist_mat)
            cont_mat = torch.from_numpy(cont_mat)
            if self.unsqueeze:
                dist_mat = dist_mat.unsqueeze(0)
                cont_mat = cont_mat.unsqueeze(0)
        return dist_mat, cont_mat
            
                