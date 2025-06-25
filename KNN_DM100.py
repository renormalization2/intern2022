import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree
import torch
from time import time
from tqdm import tqdm
from multiprocessing import Pool

def task(i):
    xyz = pos[i:i+1]
    dist, ind = tree.query(xyz, k=k)
    return np.delete(ind, [np.where(dist==0)][0])



start = time()
for num in np.arange(0,100):
    df_sphere = pd.read_csv(f'/data1/dhhyun/DM100/df_DM{num:02}.csv', index_col=0)
    df_sphere

    pos = np.column_stack((df_sphere['x'], df_sphere['y'], df_sphere['z']))

    k = 10
    knnlist = []
    
    # for i in tqdm(range(len(pos))):
    tree = KDTree(pos, leaf_size=10**7)
    # xyz = pos[i:i+1]
    # dist, ind = tree.query(xyz, k=k)
    with Pool(processes=4) as pool:
        knnlist = pool.map(task, np.arange(0,len(pos)))
    

    tuples = []
    for n, row in enumerate(knnlist):
        # for knn in row:
        index_tuple = np.column_stack((np.ones(len(row), dtype='int')*n, row))
        tuples.append(index_tuple)

    edge_index = torch.tensor(np.concatenate(tuples, axis=0), dtype=torch.long)
    
    
    np.savetxt(f'/data1/dhhyun/DM100/DM{num:02}_graph_k10.csv', edge_index, delimiter=',', fmt='%d') # int format
    
print(time()-start, 's')