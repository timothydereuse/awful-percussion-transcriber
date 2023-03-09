import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import dloader as dl
import triplet_loss_net as tln
import numpy as np

dset_fname = 'dataset_cqt_train.h5'
dset_fname_val = 'dataset_cqt_validate.h5'
model_out_path = "cqt_big_saved_model.pt"
epochs = 150
batch_size = 2500
output_dim = 10
time_since_best = 12 # for early stopping on val set

dataset = dl.TripletDrumsDataset(dset_fname, flatten=True)
dloader = DataLoader(dataset, batch_size=batch_size)
dataset_val = dl.TripletDrumsDataset(dset_fname_val, flatten=True)
dloader_val = DataLoader(dataset_val, batch_size=batch_size)

model_kwargs = [np.product(dataset.dim[1:]), output_dim]
model = tln.EmbeddingNetwork(*model_kwargs)

# model_kwargs = {'inp_dim': dataset.dim[2], 'hidden_dim': 128, 'emb_dim': 10}
# model = tln.EmbeddingNetworkLSTM(**model_kwargs)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f'created model with {total_params} params.')

criterion = torch.nn.TripletMarginLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.01, mode='triangular2', step_size_up=25, cycle_momentum=False)

val_losses = []
train_losses = []
model.train()
model.float()
for epoch in range(epochs):
    running_loss = []

    model.train()
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
    train_loss = np.mean(running_loss)
    model.eval()

    val_running_loss = []
    for step, (anchor, positive, negative) in enumerate(dloader_val):
        anchor_out = model(anchor.float())
        positive_out = model(positive.float())
        negative_out = model(negative.float())
        
        loss = criterion(anchor_out, positive_out, negative_out)

        val_running_loss.append(loss.cpu().detach().numpy())
        print(f'Epoch {epoch+1}, val {step}, val_loss {val_running_loss[-1]:.4f}')
    val_loss = np.mean(val_running_loss)

    print("Epoch: {}/{} - Loss: {:.4f} - Val Loss: {:.4f}".format(epoch+1, epochs, train_loss, val_loss))

    val_losses.append(val_loss)
    train_losses.append(train_loss)
    lowest_val_loss = min(val_losses)

    if np.all(lowest_val_loss < np.array(val_losses[-time_since_best:])):
        print("early stopping due to validation loss")
        break
    

torch.save({
    'model_args': model_kwargs,
    'model_state_dict': model.state_dict()
    }, model_out_path)

# res = (torch.load(PATH))
# new_model = tln.EmbeddingNetwork(*res['model_args'])
# new_model.eval()

