from sklearn.neighbors import KDTree
from time import time
from tqdm import tqdm
import numpy as np
import pandas as pd

df_star_sel = pd.read_csv('/data1/dhhyun/PyG/df_star_sel.csv', index_col=0)
pos = np.column_stack((df_star_sel['x'], df_star_sel['y'], df_star_sel['z']))

start = time()

k = 20
knnlist = []
tree = KDTree(pos, leaf_size=10**8)
# for xyz in pos:
for i in tqdm(range(len(pos))):
    xyz = pos[i:i+1]
    dist, ind = tree.query(xyz, k=k)
    knnlist.append(np.delete(ind, [np.where(dist==0)][0]))
    # knnlist.append(ind)
    # if i%int(len(pos)/100)==0: print(time()-start, f's elapsed, {i}th checkpoint')

t = time() - start
print('total time', f'{t/60} min' if t<60*60 else f'{t/3600} hours')

# Save
tuples = []
for n, row in enumerate(knnlist):
    # for knn in row:
    index_tuple = np.column_stack((np.ones(len(row), dtype='int')*n, row))
    tuples.append(index_tuple)
    
edge_index_array = np.concatenate(tuples, axis=0)
np.savetxt('m12f_graph_k20.csv', edge_index_array, delimiter=',', fmt='%d') # int format
