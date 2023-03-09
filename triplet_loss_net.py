import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset

class EmbeddingNetwork(nn.Module):
    def __init__(self, inp_dim, emb_dim=10):
        super(EmbeddingNetwork, self).__init__()

        l1_dim = 256
        l2_dim = 256
        l3_dim = 128

        self.fc = nn.Sequential(
            nn.Linear(inp_dim, l1_dim),
            nn.SiLU(),
            nn.BatchNorm1d(l1_dim),
            nn.Dropout(0.25),
            nn.Linear(l1_dim, l2_dim),
            nn.SiLU(),
            nn.BatchNorm1d(l2_dim),
            nn.Dropout(0.25),
            nn.Linear(l2_dim, l3_dim),
            nn.SiLU(),
            nn.BatchNorm1d(l3_dim),
            nn.Dropout(0.25),
            nn.Linear(l3_dim, emb_dim)
        )
        
    def forward(self, x):
        x = self.fc(x)
        return x

class EmbeddingNetworkLSTM(nn.Module):
    def __init__(self, inp_dim, hidden_dim=128, emb_dim=10):
        super(EmbeddingNetworkLSTM, self).__init__()
        num_layers = 2

        self.lstm = nn.LSTM(input_size=inp_dim, hidden_size=hidden_dim, num_layers=num_layers, dropout=0.2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, emb_dim)
        
    def forward(self, x):
        output, (hidden, cell) = self.lstm(x.swapaxes(1, 2))
        ret = self.linear(hidden[-2:].swapaxes(0, 1).reshape(x.shape[0], -1))
        return ret

if __name__ == '__main__':
    from dloader import TripletDrumsDataset

    dset = TripletDrumsDataset('dataset_groups.h5')
    dloader = DataLoader(dset, batch_size=50)

    model = EmbeddingNetwork(dset.dim, 32)
    model.float()

    for anchors, positives, negatives in dloader:
        asdf = model(anchors.float())
        print(asdf.shape)
