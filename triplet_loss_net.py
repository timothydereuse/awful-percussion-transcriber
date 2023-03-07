import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset

class EmbeddingNetwork(nn.Module):
    def __init__(self, inp_dim, emb_dim=64):
        super(EmbeddingNetwork, self).__init__()

        l1_dim = 256
        l2_dim = 128

        self.fc = nn.Sequential(
            nn.Linear(inp_dim, l1_dim),
            nn.SiLU(),
            nn.BatchNorm1d(l1_dim),
            nn.Dropout(0.25),
            nn.Linear(l1_dim, l2_dim),
            nn.SiLU(),
            nn.BatchNorm1d(l2_dim),
            nn.Dropout(0.25),
            nn.Linear(l2_dim, emb_dim)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x

if __name__ == '__main__':
    from dloader import TripletDrumsDataset

    dset = TripletDrumsDataset('dataset_groups.h5')
    dloader = DataLoader(dset, batch_size=50)

    model = EmbeddingNetwork(dset.dim, 32)
    model.float()

    for anchors, positives, negatives in dloader:
        asdf = model(anchors.float())
        print(asdf.shape)
