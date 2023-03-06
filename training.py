import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import dloader as dl
import triplet_loss_net as tln
import numpy as np

dset_fname = 'dataset_groups.h5'
epochs = 50

dataset = dl.TripletDrumsDataset(dset_fname)
dloader = DataLoader(dataset, batch_size=50)
model = tln.EmbeddingNetwork(dataset.dim, 32)

criterion = torch.nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.1, mode='triangular2', step_size_up=200, cycle_momentum=False)

model.train()
model.float()
for epoch in range(epochs):
    running_loss = []
    for step, (anchor, positive, negative) in enumerate(dloader):
        
        optimizer.zero_grad()
        anchor_out = model(anchor.float())
        positive_out = model(positive.float())
        negative_out = model(negative.float())
        
        loss = criterion(anchor_out, positive_out, negative_out)
        loss.backward()
        optimizer.step()
        
        running_loss.append(loss.cpu().detach().numpy())

    scheduler.step()
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))