# m12f GNN
######### ######### ######### ######### ######### ######### ######### #######79 

import os
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.datasets import GNNBenchmarkDataset


# use GPUs if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'{device}', 'available')

# Load Data & Graph
dataset = GNNBenchmarkDataset(root='/data1/dhhyun/PyG', name='PATTERN')


# Normalization
# scaler = MinMaxScaler() # axis=0 default
# scaler.fit(X) 
# X = scaler.transform(X)

# Train, Val, Test mask
trainloader = DataLoader(dataset, batch_size=32, shuffle=True)
# later consider training/test mask

# train_mask = np.zeros(len(y), dtype='int')
# num_samples = int(0.9 * len(y)) #later revise to be 8:1:1
# selected_indices = np.random.choice(train_mask, size=num_samples, replace=False)
# train_mask[selected_indices] = 1
# np.savetxt('/data1/dhhyun/train_mask.csv', train_mask, delimiter=',', fmt='%d')
# test_mask = torch.tensor(~train_mask, dtype=torch.bool)
# train_mask = torch.tensor(train_mask, dtype=torch.bool)

# Data Object
# X = torch.tensor(X, dtype=torch.float)
# y = torch.tensor(y, dtype=torch.float) # not int32 #################### originally torch.long for class identification
# data = Data(x=X, edge_index=edge_index, y=y,
#             train_mask=train_mask, test_mask=test_mask)#, pos=pos_sub # no need; we have edges already


# GCN Layer
num_node_features = dataset[0].num_node_features  #data.num_node_features
num_hidden = 3 ###########################

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(num_node_features, num_hidden)
        self.conv2 = GCNConv(num_hidden, 2) #data.num_classes)
        self.conv3 = GCNConv(2, 1) ############### Added

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) # self.training?
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        # return F.log_softmax(x, dim=1)
        m = torch.nn.Sigmoid()
        return m(x)


# Train
model = GCN().to(device)
# dataset = dataset.to(device) ### doesn't work
# data = data.to(device) ######################################################## Where for batches?
criterion = torch.nn.BCELoss() ###############
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
start = time()

loss_ = []
n = len(trainloader) # for n batches
# n = 1

model.train()
for epoch in tqdm(range(200)):
    running_loss = 0.0
    
    for data in trainloader:
        data = data.to(device) ###########################################################
        optimizer.zero_grad()
        out = model(data)
        # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        ###############
        # out_T = out[data.train_mask].t()[0]
        # loss = criterion(out_T, data.y[data.train_mask]) ###############
        out_T = out.t()[0]
        y_true = torch.tensor(data.y.clone().detach(), dtype=torch.float)
        loss = criterion(out_T, y_true)
        ##############
        loss.backward()
        optimizer.step()

        running_loss += loss.item() # for the case using trainloader inside a for loop
    loss_.append(running_loss/n) # MSE(Mean Squared Error)
    
t = (time()-start)
print('total time', f'{t/60} min' if t>60 else f'{t} sec')

np.savetxt('/data1/dhhyun/loss.csv', loss_, delimiter=',')
torch.save(model.state_dict(), '/data1/dhhyun/model.pth')
    
    
# Evaluation
model.eval()
# pred = model(data).argmax(dim=1)
pred = model(data) #####################

pred_save = pred.cpu().detach().numpy()
np.savetxt('/data1/dhhyun/pred.csv', pred_save, delimiter=',')#, fmt='%d')
np.savetxt('/data1/dhhyun/pred_true.csv', data.y.cpu().clone().detach(), delimiter=',')#, fmt='%d')

# thresh = np.percentile(pred_save, 99)
# pred_int = np.where(pred_save <= thresh, 0, 1)
# correct = (pred_int[data.test_mask] == data.y[data.test_mask]).sum()
# acc = int(correct) / int(data.test_mask.sum())
# print(f'Accuracy: {acc:.4f}')