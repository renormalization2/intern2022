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

# use GPUs if available
if torch.cuda.is_available():
    print("CUDA Available")
    device = torch.device('cuda')
else:
    print('CUDA Not Available')
    device = torch.device('cpu')
# One liner:
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Data & Graph
df_star_sel = pd.read_csv('/data1/dhhyun/PyG/df_star_sel.csv', index_col=0)
X = np.column_stack((df_star_sel['r'], df_star_sel['theta'], df_star_sel['phi'],
    df_star_sel['v_r'], df_star_sel['v_theta'], df_star_sel['v_phi']))
y = np.array(df_star_sel['group'] == 'halo_associated').astype('int')

edge_index = np.genfromtxt('/data1/dhhyun/m12f_graph_k10.csv', delimiter=',', dtype='int')
edge_index = torch.tensor(edge_index, dtype=torch.long)
edge_index = edge_index.t().contiguous()

# Normalization
scaler = MinMaxScaler() # axis=0 default
scaler.fit(X) 
X = scaler.transform(X)

# Train, Val, Test mask
train_mask = np.zeros(len(y), dtype='int')
num_samples = int(0.9 * len(y)) #later revise to be 8:1:1
selected_indices = np.random.choice(train_mask, size=num_samples, replace=False)
train_mask[selected_indices] = 1
np.savetxt('/data1/dhhyun/train_mask.csv', train_mask, delimiter=',', fmt='%d')
test_mask = torch.tensor(~train_mask, dtype=torch.bool)
train_mask = torch.tensor(train_mask, dtype=torch.bool)

# Data Object
X = torch.tensor(X, dtype=torch.float)
y = torch.tensor(y, dtype=torch.long) # not int32
data = Data(x=X, edge_index=edge_index, y=y,
            train_mask=train_mask, test_mask=test_mask)#, pos=pos_sub # no need; we have edges already


# GCN Layer
num_hidden = 4 #6 #2 #3

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(data.num_node_features, num_hidden)
        # self.conv3 = GCNConv(num_hidden, num_hidden)
        self.conv2 = GCNConv(num_hidden, 2) #data.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        # x = self.conv3(x, edge_index)
        # x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


# Train
model = GCN().to(device)
data = data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
start = time()

loss_ = []
# n = len(trainloader) # for n batches
n = 1

model.train()
for epoch in tqdm(range(200)):
    running_loss = 0.0 
    
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    # loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
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
pred = model(data).argmax(dim=1)
correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
acc = int(correct) / int(data.test_mask.sum())
print(f'Accuracy: {acc:.4f}')

pred_save = pred.cpu()
np.savetxt('/data1/dhhyun/pred.csv', pred_save, delimiter=',', fmt='%d')
