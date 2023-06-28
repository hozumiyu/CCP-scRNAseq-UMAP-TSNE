# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:41:21 2023

@author: Yuta
"""

import sys, os
import numpy as np
from algorithm.auxilary import load_X, load_y, preprocess_data, makeFolder
from algorithm.FeaturePartition import writeIndex, divide_features
from algorithm.reduction import computeCCP, computeUMAP
from sklearn.preprocessing import StandardScaler
from algorithm.clustering import runKM

data = sys.argv[1]
n_components = int(sys.argv[2])
cutoff = float(sys.argv[3])
random_state = int(sys.argv[4])
scaling = sys.argv[5]

ccp_path = './features/%s_features/%s_ccp_features/'%(data, data); makeFolder(ccp_path)
partition_path = './features/%s_features/%s_partition/'%(data, data); makeFolder(partition_path)
file = '%s_c%.1f_n%d_s%d'%(data, cutoff, n_components, random_state)

print('Starting >>>>>>>>>>')
print('Data :%s, n_components: %d, cutoff:%.1f, random_state:%d, scaling:%s'%(data, n_components, cutoff, random_state, scaling))
X = load_X(data); y = load_y(data)
if data == 'GSE57249':
    X = np.log10(X+1).T
else:
    X, y = preprocess_data(X,y)
variance = np.var(X, axis = 0)
index = np.where(variance > 1e-6)[0]
X = X[:, index]

if not os.path.exists(ccp_path + file + '.npy') or not os.path.exists(partition_path + file + '_partition.txt'):
    print('computing ccp >>>>>>>>>>')
    index_features = divide_features(X, n_components, cutoff = cutoff, random_state = random_state)
    writeIndex(outfile = partition_path + file + '_partition.txt', index_features = index_features)
    X_ccp = computeCCP(X, n_components, index_features)
    np.save(ccp_path + file + '.npy', X_ccp)
    print('finished computing ccp >>>>>>>>>>')
else:
    print('already computed ccp >>>>>>>>>>')
    X_ccp = np.load(ccp_path + file + '.npy')
    
if scaling == 'std':
    X_ccp = StandardScaler().fit_transform(X_ccp)


print('Compute Kmeans clustering for CCP >>>>>>>>>>')
outpath_ccp_km = './results_km/%s_results_km/'%(data)
if not os.path.exists(outpath_ccp_km + file + '_km.npy'):
    makeFolder(outpath_ccp_km)
    ari = runKM(X_ccp, y)
    np.save(outpath_ccp_km + file + '_km.npy', ari)
else:
    print('Already computed Kmeans clustering for CCP >>>>>>>>>>')
    ari = np.load(outpath_ccp_km + file + '_km.npy')
print('CCP kmeans result: ', np.mean(ari))




umap_path = './features/%s_features/%s_ccp_umap_features/'%(data, data); makeFolder(umap_path)
if not os.path.exists(umap_path + file + '_umap.npy'):
    print('Computing UMAP >>>>>>>>>>')
    X_umap = computeUMAP(X_ccp, n_neighbors = 15)
    np.save(umap_path + file + '_umap.npy', X_umap)
else:
    print('Already omputed t-SNE >>>>>>>>>>')
    X_umap = np.load(umap_path + file + '_umap.npy')


X_umap = StandardScaler().fit_transform(X_umap)
print('Compute Kmeans clustering for CCP-assisted t-SNE >>>>>>>>>>')
outpath_umap_km = './results_km/%s_results_km/'%(data)
if not os.path.exists(outpath_umap_km + file + '_umap_km.npy'):
    makeFolder(outpath_umap_km)
    ari = runKM(X_umap, y)
    np.save(outpath_umap_km + file + '_umap_km.npy', ari)
else:
    print('Already computed Kmeans clustering for CCP >>>>>>>>>>')
    ari = np.load(outpath_umap_km + file + '_umap_km.npy')
print('CCP-assisted UMAP kmeans result: ', np.mean(ari))
    
