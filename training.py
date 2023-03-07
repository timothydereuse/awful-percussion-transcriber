import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import dloader as dl
import triplet_loss_net as tln
import numpy as np

dset_fname = 'dataset_groups_big.h5'
epochs = 200
batch_size = 5000
output_dim = 10

dataset = dl.TripletDrumsDataset(dset_fname)
dloader = DataLoader(dataset, batch_size=batch_size)
model = tln.EmbeddingNetwork(dataset.dim, output_dim)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'created model with {total_params} params.')

criterion = torch.nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular2', step_size_up=30, cycle_momentum=False)

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
        print(f'Epoch {epoch+1}, batch {step}, loss {running_loss[-1]}')

    scheduler.step()
    print("Epoch: {}/{} - Loss: {:.4f}".format(epoch+1, epochs, np.mean(running_loss)))

PATH = "saved_model.pt"

torch.save({
    'model_args': (dataset.dim, output_dim),
    'model_state_dict': model.state_dict()
    }, PATH)

# res = (torch.load(PATH))
# new_model = tln.EmbeddingNetwork(*res['model_args'])
# new_model.eval()

