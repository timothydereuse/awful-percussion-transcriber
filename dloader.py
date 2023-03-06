import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import h5py

class TripletDrumsDataset(IterableDataset):
    def __init__(self, dset_fname):
        self.dset_fname = dset_fname
        self.f = h5py.File(self.dset_fname, 'r')

        self.keys = list(self.f.keys())
        self.numel = sum([self.f[k].shape[0] for k in self.keys])
        self.dim = self.f[self.keys[0]].shape[1]

    def __iter__(self):
        f = self.f
        for k in self.keys:
            dset = f[k]

            other_keys = list(self.keys)
            other_keys.remove(k)

            inds = np.arange(dset.shape[0])
            for ind in inds:
                anchor = f[k][ind].astype(float)
                pos_ind = np.random.choice(np.concatenate([inds[:ind], inds[ind+1:]]))
                pos = f[k][pos_ind].astype(float)
                neg_k = np.random.choice(other_keys)
                neg_ind = np.random.choice(f[neg_k].shape[0])
                neg = f[neg_k][neg_ind].astype(float)
                yield anchor, pos, neg


