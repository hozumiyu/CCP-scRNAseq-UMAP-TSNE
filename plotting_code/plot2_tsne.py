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
#GSE67835
data, method, scaling, cutoff, n_components = 'GSE67835' , 'umap', 'std' ,'0.7','300'
row = 0
y = load_y(data)
mapping = {0: 0, 1:1, 2:2, 3:6 ,4:7, 5:5, 6:3, 7:4}
y = remapLabels(y, mapping)
#legend = ['OPC', 'Astrocytes', 'Endothelial', 'Microglia', 'Neurons', 'Oligodendrocytes',  'Fetal Quiescent', 'Fetal Replicating']
legend = ['OPC', 'Atro', 'Endo', 'Microglia', 'Neurons', 'Oligo',  'Fetal-Q', 'Fetal-R']


inpath = '../features2/%s_features/'%( data)
file = '%s_ccp_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])

legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE67835', rotation=90)

###########################################################################
#GSE82187
data, method, scaling, cutoff, n_components = 'GSE82187' , 'umap', 'raw' ,'0.7','150'
row = 1
y = load_y(data)
mapping = {0: 0, 1:6, 2:7, 3:3 ,4:4, 6:5, 8:1, 9:2}
y = remapLabels(y, mapping)
#legend = ['Astrocytes', 'Macrophage', 'Microglia', 'Neuron', 'Oligodendrocytes', 'Vascular', 'Ependymal Cilated', 'Ependymal Secreted']
legend = ['Astro', 'Macro', 'Microglia', 'Neuron', 'Oligo', 'Vascular', 'Ependy-C', 'Ependy-S']



inpath = '../features2/%s_features/'%( data)
file = '%s_ccp_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])

legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE82187', rotation=90)

###########################################################################
#GSE84133human4
data, method, scaling, cutoff, n_components = 'GSE84133human4' , 'umap', 'raw' ,'0.6','150'
row = 2
y = load_y(data)
#legend = ['Activated Stellate', r'Alpha', 'Beta', 'Delta', 'Ductal', 'Gamma']
legend = ['Activated Stellate', r'$\alpha$', r'$\beta$', r'$\delta$', 'Ductal', r'$\gamma$']


inpath = '../features2/%s_features/'%( data)
file = '%s_ccp_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])


legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE84133 H', rotation=90)

###########################################################################
#GSE84133mouse1
data, method, scaling, cutoff, n_components = 'GSE84133mouse1' , 'umap', 'raw' ,'0.8','250'
row = 3
y = load_y(data)
legend = [r'$\beta$', r'$\delta$', 'Ductal', 'Endothelial', 'Macrophage', 'Quiescent Stellate']


inpath = '../features2/%s_features/'%( data)
file = '%s_ccp_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 0] = plot(X, y, ax[row, 0])


file = '%s_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 1] = plot(X, y, ax[row, 1])

file = '%s_pca_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 2] = plot(X, y, ax[row, 2])

file = '%s_nmf_tsne.npy'%(data)
X = np.load(inpath + file)
ax[row, 3] = plot(X, y, ax[row, 3])

legend = ax[row, 0].legend(legend, handletextpad=0., 
         bbox_to_anchor=(0, 0.03, 4, 0.03),fancybox=False, shadow=False, ncol=20, mode="expand")
legend.get_frame().set_alpha(0)
for legend_handle in legend.legend_handles:
    legend_handle._sizes = [15]
ax[row, 0].set_ylabel('GSE84133 M', rotation=90)


fig.subplots_adjust(wspace=0)

ax[0, 0].set_title('CCP t-SNE')
ax[0, 1].set_title('t-SNE')
ax[0, 2].set_title('NMF t-SNE')
ax[0, 3].set_title('PCA t-SNE')
size = 10
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
fig.savefig(outpath + 'results2_tsne.png', bbox_inches='tight', dpi = 500)
'''
data_vec = []
data_vec.append(['GSE75748cell' , 250])
data_vec.append(['GSE75748time' , 300])
data_vec.append( ['GSE84133human3', 150])
data_vec.append( ['GSE94820', 300])
cutoff = 0.7

fig, ax = plt.subplots(nrows=4, ncols = 4, figsize = (16, 18))
outpath = './figures/'; makeFolder(outpath)

for row, data_param in enumerate(data_vec):
    
    data, n_components = data_param
    y = load_y(data)
    
    
    inpath = '../CCP_secondary_cutoff_main/features/%s_features/%s_ccp_umap_scaled_cutoff/'%(data, data)
    file = '%s_ccp_n%d_c%.1f_umap_nn%d_scaled_cutoff.npy'%(data, n_components, cutoff, 15)
    X = np.load(inpath + file)
    ax[row, 0] = plot(X, y, ax[row, 0])
    
    inpath = '../CCP_secondary_cutoff_main/features/%s_features/%s_umap_cutoff/'%(data, data)
    file = '%s_umap_nn%d_cutoff.npy'%(data, 15)
    X = np.load(inpath + file)
    ax[row, 1] = plot(X, y, ax[row, 1])
    
    inpath = '../CCP_secondary_cutoff_main/features/%s_features/%s_ccp_tsne_scaled_cutoff/'%(data, data)
    file = '%s_ccp_n%d_c%.1f_tsne_p%d_scaled_cutoff.npy'%(data, n_components, cutoff, 30)
    X = np.load(inpath + file)
    ax[row, 2] = plot(X, y, ax[row, 2])
    
    inpath = '../CCP_secondary_cutoff_main/features/%s_features/%s_tsne_cutoff/'%(data, data)
    file = '%s_tsne_p%d_cutoff.npy'%(data, 30)
    title = 'tsne'
    X = np.load(inpath + file)
    ax[row, 3] = plot(X, y, ax[row, 3])
    
    legend = extractLegend(data, y)
    ax[row, 0].legend(legend,
             bbox_to_anchor=(0, -0.01, 4, -0.01),fancybox=False, shadow=False, ncol=20, mode="expand")
    plt.subplots_adjust(wspace=0)
    if data =='GSE84133human3':
        data = 'GSE84133 h3'
    elif data == 'GSE84133human4':
        data = 'GSE84133 h4'
    elif data == 'GSE84133mouse1':
        data = 'GSE84133 m1'
    ax[row, 0].set_ylabel(data, rotation=90)
    

ax[0, 0].set_title('CCP UMAP')
ax[0, 1].set_title('UMAP')
ax[0, 2].set_title('CCP t-SNE')
ax[0, 3].set_title('t-SNE')
plt.tight_layout()
fig.savefig(outpath + 'results1.jpg')
'''