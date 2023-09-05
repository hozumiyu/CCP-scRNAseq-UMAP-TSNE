# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 23:20:17 2023

@author: yutah
"""

import numpy as np
import pandas as pd
import os
from algorithm.auxilary import load_X, load_y, preprocess_data, makeFolder
from algorithm.CCP_main import divide_features
from sklearn.manifold import TSNE
from algorithm.CCP.ccp import CCP
import matplotlib.pyplot as plt
from umap import UMAP

data_process_path = './SingleCellDataProcess/'   #change this line if you have a different path for the data processing
data_path = './data/'  #location of where the processed raw data will be saved

def plot(X, y, ax):
    color = [plt.cm.tab10(l) for l in range(10)]
    color[7] = color[-1]
    labels = np.unique(y)
    labels.sort()
    for idx, l in enumerate(labels):
        index = np.where(y == l)[0]
        ax.scatter(X[index, 0], X[index, 1], color = color[idx], s =3 )
    ax.set_xticks([])
    ax.set_yticks([])
    return ax

def subSample(data, inpath, unique_labels):
    X = load_X(data, data_path, data_process_path)
    inpath = inpath + data + '/'
    y_record = pd.read_csv(inpath + '%s_labels.csv'%(data))
    y_labels = np.array(y_record['Cell type'])
    
    index = []
    for idx, cell in enumerate(y_labels):
        if cell in unique_labels:
            index.append(idx)
    index = np.array(index)
    X = X.T
    return X[index, :], y_labels[index]

def map_labels(y_labels, mapping):
    y_new = np.zeros(y_labels.shape[0])
    for idx in range(y_labels.shape[0]):
        y_new[idx] = mapping[y_labels[idx]]
    return y_new
    
    

unique_labels = [ 'alpha', 'beta', 'delta', 'gamma', 'ductal']
mapping = { u:i for i, u in enumerate(unique_labels)}
outpath = './features/subsample/'; makeFolder(outpath)

for idx in range(1, 5):
    print('Patient', idx)
    if not os.path.exists(outpath + 'GSE84133_ss%d_X_ccp.npy'%(idx)):
        data = 'GSE84133human%d'%(idx)
        X_train, y_train_temp = subSample(data, data_path, unique_labels)
        y_train =  map_labels(y_train_temp, mapping)
        X_train = np.log10(1+X_train)
        index_features = divide_features(X_train, n_components=150, cutoff = 0.7, random_state = 1)
        myCCP = CCP(n_components = len(index_features))
        myCCP.fit(X_train, index_features)
        X_ccp_train = myCCP.transform(X_train)
        y = [y_train]
        X_ccp = [X_ccp_train]
        for idx2 in range(1,5):
            if idx != idx2:
                data = 'GSE84133human%d'%(idx2)
                X_test, y_test_temp = subSample(data, data_path, unique_labels)
                y_test =  map_labels(y_test_temp, mapping)
                y.append(y_test)
                X_test = np.log10(1+X_test)
                X_ccp_test = myCCP.transform(X_test)
                X_ccp.append(X_ccp_test)
        X_ccp = np.concatenate(X_ccp)
        y = np.concatenate( y)
        np.save(outpath + 'GSE84133_ss%d_X_ccp.npy'%(idx), X_ccp)
        np.save(outpath + 'GSE84133_ss%d_y.npy'%(idx), y)
    else:
        X_ccp = np.load(outpath + 'GSE84133_ss%d_X_ccp.npy'%(idx))
        y = np.load(outpath + 'GSE84133_ss%d_y.npy'%(idx))
        
    if  not os.path.exists(outpath + 'GSE84133_ss%d_X_umap.npy'%(idx)):
        myUMAP = UMAP(random_state = 1, n_neighbors = 10)
        X_umap = myUMAP.fit_transform(X_ccp)
        np.save(outpath + 'GSE84133_ss%d_X_umap.npy'%(idx), X_umap)
    if  not os.path.exists(outpath + 'GSE84133_ss%d_X_tsne.npy'%(idx)):
        myTSNE = TSNE(random_state = 1, perplexity = 5)
        X_tsne = myTSNE.fit_transform(X_ccp)
        np.save(outpath + 'GSE84133_ss%d_X_tsne.npy'%(idx), X_tsne)



fig, ax = plt.subplots(2, 4, figsize = (8.5, 8.5/2))
for idx in range(1, 5):
    y = np.load(outpath + 'GSE84133_ss%d_y.npy'%(idx))
    X_umap = np.load(outpath + 'GSE84133_ss%d_X_umap.npy'%(idx))
    
    plot(X_umap, y, ax[0, idx-1])
    
    X_tsne = np.load(outpath + 'GSE84133_ss%d_X_tsne.npy'%(idx))
    
    plot(X_tsne, y, ax[1, idx-1])

legend = ax[0, 0].legend(unique_labels, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
    
legend = ax[1, 0].legend(unique_labels, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)

for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[0, 0].set_ylabel('CCP UMAP', rotation=90)

for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[1, 0].set_ylabel('CCP t-SNE', rotation=90)


ax[0, 0].set_title('Patient 1')
ax[0, 1].set_title('Patient 2')
ax[0, 2].set_title('Patient 3')
ax[0, 3].set_title('Patient 4')
size = 10
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title

