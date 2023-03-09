import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
import h5py

class TripletDrumsDataset(IterableDataset):
    def __init__(self, dset_fname, flatten=True):
        self.dset_fname = dset_fname
        self.f = h5py.File(self.dset_fname, 'r')

        self.keys = list(self.f.keys())
        self.numel = sum([self.f[k].shape[0] for k in self.keys])
        self.dim = self.f[self.keys[0]].shape
        self.flatten = flatten

    def __iter__(self):
        f = self.f
        new_keys = list(self.keys)
        np.random.shuffle(new_keys)

        for k in new_keys:
            dset = f[k]

            other_keys = list(new_keys)
            other_keys.remove(k)

            inds = np.arange(dset.shape[0])
            for ind in inds:
                anchor = f[k][ind].astype(float)
                pos_ind = np.random.choice(np.concatenate([inds[:ind], inds[ind+1:]]))
                pos = f[k][pos_ind].astype(float)
                neg_k = np.random.choice(other_keys)
                neg_ind = np.random.choice(f[neg_k].shape[0])
                neg = f[neg_k][neg_ind].astype(float)
                
                if self.flatten:
                    anchor = anchor.ravel()
                    pos = pos.ravel()
                    neg = neg.ravel()
                
                yield anchor, pos, neg


