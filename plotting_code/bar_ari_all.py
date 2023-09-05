# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:32:29 2023

@author: Yuta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

## Construct ARI barplots for each data. First bar plot is for data set 'GSE75748cell', 'GSE75748time', 'GSE57249', 'GSE94820'
## Second bar plot is for datasets 'GSE67835', 'GSE82187', 'GSE84133human4', 'GSE84133mouse1'
# Extrac the result from ../results/ folder

inpath ='../results/'
#####
data_vec = ['GSE75748cell', 'GSE75748time', 'GSE57249', 'GSE94820']

size = 10
width = 0.25
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title


ccp_umap_ari = []
ccp_tsne_ari = []
umap_ari = []
tsne_ari = []
nmf_umap_ari =[]
nmf_tsne_ari = []
pca_umap_ari = []
pca_tsne_ari = []
for data in data_vec:
    record = pd.read_csv(inpath + '%s_results.csv'%(data))
    ccp_umap_ari.append(record['CCP UMAP'][0])
    ccp_tsne_ari.append(record['CCP TSNE'][0])
    umap_ari.append(record['UMAP'][0])
    tsne_ari.append(record['TSNE'][0])
    nmf_umap_ari.append(record['NMF UMAP'][0])
    nmf_tsne_ari.append(record['NMF TSNE'][0])
    pca_umap_ari.append(record['PCA UMAP'][0])
    pca_tsne_ari.append(record['PCA TSNE'][0])
fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(4) * 2.75
ax.bar(x, ccp_umap_ari, width, label = 'CCP UMAP')
ax.bar(x+0.25, ccp_tsne_ari, width, label = 'CCP t-SNE')
ax.bar(x+0.5, umap_ari, width, label = 'UMAP')
ax.bar(x+0.75, tsne_ari, width, label = 't-SNE')
ax.bar(x+1.0, nmf_umap_ari, width, label = 'NMF UMAP')
ax.bar(x+1.25, nmf_tsne_ari, width, label = 'NMF UMAP')
ax.bar(x+1.5, pca_umap_ari, width, label = 'PCA UMAP')
ax.bar(x+1.75, pca_tsne_ari, width, label = 'PCA t-SNE')



ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_ylim([0, 1.2])
ax.set_xticks(x + 0.87, ['GSE75748 cell', 'GSE75748 time', 'GSE57249', 'GSE94820'])
legend_name = ['CCP UMAP', 'CCP t-SNE', 'UMAP', 't-SNE', 'NMF UMAP' , 'NMF t-SNE', 'PCA UMAP', 'PCA t-SNE']
ax.legend(legend_name, handletextpad=0, ncol=4, loc = 'upper left', framealpha = 0)
plt.savefig('../figures/results1_ari.png',  bbox_inches='tight', dpi = 500)

#######################################################################################
    
data_vec = ['GSE67835', 'GSE82187', 'GSE84133human4', 'GSE84133mouse1']

size = 10
width = 0.25
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title





ccp_umap_ari = []
ccp_tsne_ari = []
umap_ari = []
tsne_ari = []
nmf_umap_ari =[]
nmf_tsne_ari = []
pca_umap_ari = []
pca_tsne_ari = []
for data in data_vec:
    record = pd.read_csv(inpath + '%s_results.csv'%(data))
    ccp_umap_ari.append(record['CCP UMAP'][0])
    ccp_tsne_ari.append(record['CCP TSNE'][0])
    umap_ari.append(record['UMAP'][0])
    tsne_ari.append(record['TSNE'][0])
    nmf_umap_ari.append(record['NMF UMAP'][0])
    nmf_tsne_ari.append(record['NMF TSNE'][0])
    pca_umap_ari.append(record['PCA UMAP'][0])
    pca_tsne_ari.append(record['PCA TSNE'][0])
fig = plt.figure()
ax = fig.add_subplot(111)

x = np.arange(4) * 2.75
ax.bar(x, ccp_umap_ari, width, label = 'CCP UMAP')
ax.bar(x+0.25, ccp_tsne_ari, width, label = 'CCP t-SNE')
ax.bar(x+0.5, umap_ari, width, label = 'UMAP')
ax.bar(x+0.75, tsne_ari, width, label = 't-SNE')
ax.bar(x+1.0, nmf_umap_ari, width, label = 'NMF UMAP')
ax.bar(x+1.25, nmf_tsne_ari, width, label = 'NMF UMAP')
ax.bar(x+1.5, pca_umap_ari, width, label = 'PCA UMAP')
ax.bar(x+1.75, pca_tsne_ari, width, label = 'PCA t-SNE')



ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticks(x + 0.87, ['GSE67835', 'GSE82187', 'GSE84133 H', 'GSE84133 M'])
legend_name = ['CCP UMAP', 'CCP t-SNE', 'UMAP', 't-SNE', 'NMF UMAP' , 'NMF t-SNE', 'PCA UMAP', 'PCA t-SNE']
ax.legend(legend_name, handletextpad=0.1, ncol=2, loc = 'upper left', framealpha = 0)
ax.set_ylim([0, 1])

plt.savefig('../figures/results2_ari.png',  bbox_inches='tight', dpi = 500)



