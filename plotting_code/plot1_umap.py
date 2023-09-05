#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  4 10:53:28 2023

@author: yutaho
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


data_path = '../data/'
def makeFolder(path):
    try: 
        os.makedirs(path)
    except:
        return    
    return 
def load_y_nodrop(data, data_path = data_path):
    inpath = data_path + '%s/'%(data)
    y_record = pd.read_csv(inpath + '%s_labels.csv'%(data))
    y = np.array(list(y_record['Label'])).astype(int)
    return y

def load_y(data, data_path = data_path):
    inpath = data_path + '%s/'%(data)
    y_record = pd.read_csv(inpath + '%s_labels.csv'%(data))
    y = np.array(list(y_record['Label'])).astype(int)
    labels = np.unique(y)
    good_index = []
    for l in labels:
        index = np.where(y == l)[0]
        if index.shape[0] > 15:
            good_index.append(index)
        else:
            print('label %d removed'%(l))
    good_index = np.concatenate(good_index)
    good_index.sort()
    y = y[good_index]
    return y[good_index]
    return y

def drop_sample(y):
    labels = np.unique(y)
    good_index = []
    for l in labels:
        index = np.where(y == l)[0]
        if index.shape[0] > 15:
            good_index.append(index)
        else:
            print('label %d removed'%(l))
    good_index = np.concatenate(good_index)
    good_index.sort()
    
    return y[good_index]

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

def extractLegend(data, y):
    labels = np.unique(y)
    labels.sort()
    inpath = '../../../data/%s/'%(data)
    file = open(inpath + '%s_full_labeldict.txt'%(data), 'r')
    lines = file.readlines()
    file.close()
    mapping = eval(lines[0])
    legend = []
    for l in labels:
        legend.append(mapping[l])
    return legend

def remapLabels(y, mapping):
    new_y = np.zeros(y.shape[0])
    for i in range(y.shape[0]):
        new_y[i] = mapping[y[i]]
    return new_y






fig, ax = plt.subplots(nrows=4, ncols = 4, figsize = (8.5, 8.5))
outpath = './figures/'; makeFolder(outpath)
###########################################################################
#GSE75748cell
data = 'GSE75748cell' 
row = 0
y = load_y(data)
mapping = {0: 0, 1:1, 2:5, 3:6 ,4:4, 5:2, 6:3}
y = remapLabels(y, mapping)
legend = ['DEC', 'EC',  'NPC', 'TB',  'HFF', 'H1', 'H9' ]


inpath = '../features2/%s_features/'%( data)
file = '%s_ccp_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])

legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE75748 cell', rotation=90)

###########################################################################

#GSE75748time
data = 'GSE75748time' 
row = 1
y = load_y(data)
legend = ['0hr', '12hr', '24hr', '36hr', '72hr', '96hr']

inpath = '../features2/%s_features/'%( data)

file = '%s_ccp_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])


legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE75748 time', rotation=90)

###########################################################################

#GSE57249
data = 'GSE57249'
row = 2
y = load_y_nodrop(data)
legend = ['Zygote', '2 Cell', '4 Cell']
#legend = ['Acinar', 'Alpha', 'Beta', 'Delta', 'Gamma', 'Ductal', 'Endothelial', 'Activated stel', 'Quiescent ste']

inpath = '../features2/%s_features/'%( data)

file = '%s_ccp_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])


legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE57249', rotation=90)


###########################################################################
#GSE94820
data = 'GSE94820'
row = 3
y = load_y(data)
legend = ['CD141', 'CD1C', 'Double Negative', 'Mono', 'pDC']

inpath = '../features2/%s_features/'%( data)

file = '%s_ccp_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_umap.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])


legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE94820', rotation=90)


fig.subplots_adjust(wspace=0)

ax[0, 0].set_title('CCP UMAP')
ax[0, 1].set_title('UMAP')
ax[0, 2].set_title('NMF UMAP')
ax[0, 3].set_title('PCA UMAP')
size = 10
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
fig.savefig(outpath + 'results1_umap.png', bbox_inches='tight', dpi = 500)
