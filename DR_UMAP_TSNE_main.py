# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:41:21 2023

@author: Yuta
"""

import sys, os
import numpy as np
from algorithm.auxilary import load_X, load_y, preprocess_data, makeFolder
from sklearn.manifold import TSNE
from umap import UMAP
from algorithm.clustering import runKM
from sklearn.preprocessing import StandardScaler

data = sys.argv[1]
random_state = int(sys.argv[2])

umap_path = './features/%s_features/%s_umap_features/'%(data, data); makeFolder(umap_path)

X = load_X(data); y = load_y(data)
if data == 'GSE57249':
    X = np.log10(X+1).T
else:
    X, y = preprocess_data(X,y)




file = '%s_umap_s%d'%(data, random_state)
if not os.path.exists(umap_path + file + '.npy'):
    myUMAP = UMAP(random_state = random_state)
    X_umap = myUMAP.fit_transform(X)
    np.save(umap_path + file + '.npy', X_umap)
else:
    X_umap = np.load(umap_path + file + '.npy')

X_umap = StandardScaler().fit_transform(X_umap)
print('Compute Kmeans clustering for UMAP >>>>>>>>>>')
outpath_umap_km = './results_km/%s_results_km/'%(data)
if not os.path.exists(outpath_umap_km + file + '_km.npy'):
    makeFolder(outpath_umap_km)
    ari = runKM(X_umap, y)
    np.save(outpath_umap_km + file + '_km.npy', ari)
else:
    print('Already computed Kmeans clustering for CCP >>>>>>>>>>')
    ari = np.load(outpath_umap_km + file + '_km.npy')
print('UMAP kmeans result: ', np.mean(ari))
    


X = load_X(data); y = load_y(data)
if data == 'GSE57249':
    X = np.log10(X+1).T
else:
    X, y = preprocess_data(X,y)
tsne_path = './features/%s_features/%s_tsne_features/'%(data, data); makeFolder(tsne_path)
file = '%s_tsne_s%d'%(data, random_state)
if not os.path.exists(tsne_path + file + '.npy'):
    myTSNE = TSNE(random_state = random_state)
    X_tsne = myTSNE.fit_transform(X)
    np.save(tsne_path + file + '.npy', X_tsne)
else:
    X_tsne = np.load(tsne_path + file + '.npy')
        
        
X_tsne = StandardScaler().fit_transform(X_tsne)
print('Compute Kmeans clustering for t-SNE >>>>>>>>>>')
outpath_tsne_km = './results_km/%s_results_km/'%(data)
if not os.path.exists(outpath_tsne_km + file + '_km.npy'):
    makeFolder(outpath_tsne_km)
    ari = runKM(X_tsne, y)
    np.save(outpath_tsne_km + file + '_km.npy', ari)
else:
    print('Already computed Kmeans clustering for t-SNE >>>>>>>>>>')
    ari = np.load(outpath_tsne_km + file + '_km.npy')
print(ari)
print('t-SNE kmeans result: ', np.mean(ari))