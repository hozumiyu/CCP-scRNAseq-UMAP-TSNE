# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 10:32:29 2023

@author: Yuta
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Generate Average ARI values across all the test for CCP asssited UMAP, CCP assisted tSNE, UMAP and tSNE

data_vec = ['GSE67835', 'GSE82187', 'GSE84133human4', 'GSE84133mouse1', 'GSE75748cell', 'GSE75748time', 'GSE94820', 'GSE57249']

size = 10
width = 0.125
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title


inpath ='../results/'


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

ccp_umap_ari = np.mean(ccp_umap_ari)    
ccp_tsne_ari = np.mean(ccp_tsne_ari)

umap_ari = np.mean(umap_ari)
tsne_ari = np.mean(tsne_ari)

nmf_umap_ari = np.mean(nmf_umap_ari)
nmf_tsne_ari = np.mean(nmf_tsne_ari)

pca_umap_ari = np.mean(pca_umap_ari)
pca_tsne_ari = np.mean(pca_tsne_ari)

fig = plt.figure(figsize = (4.5, 2.5))
ax = fig.add_subplot(111)

x = np.arange(4) * 1
ax.bar( 0, ccp_umap_ari, width, label = 'CCP UMAP')
ax.bar(0.15, ccp_tsne_ari, width, label = 'CCP t-SNE')
ax.bar(0.3, umap_ari, width, label = 'UMAP')
ax.bar(0.45, tsne_ari, width, label = 't-SNE')
#ax.bar(1.0, nmf_umap_ari, width, label = 'NMF UMAP')
#ax.bar(1.25, nmf_tsne_ari, width, label = 'NMF UMAP')
#ax.bar(1.5, pca_umap_ari, width, label = 'PCA UMAP')
#ax.bar(1.75, pca_tsne_ari, width, label = 'PCA t-SNE')


ax.set_ylabel('Mean ARI', rotation=90)
ax.set_yticks([0.4, 0.6, 0.8])
ax.set_ylim(0.4, 0.8)
ax.set_xticks( [0,0.15,0.3,0.45], ['CCP UMAP', 'CCP t-SNE', 'UMAP', 't-SNE'])
ax.text(-0.035, 0.01 + ccp_umap_ari, '%.3f'%(ccp_umap_ari))
ax.text(0.115, 0.01 + ccp_tsne_ari, '%.3f'%(ccp_tsne_ari))
ax.text(0.265, 0.01 + umap_ari, '%.3f'%(umap_ari))
ax.text(0.415, 0.01 + tsne_ari, '%.3f'%(tsne_ari))
#legend_name = ['CCP UMAP', 'CCP t-SNE', 'UMAP', 't-SNE', 'NMF UMAP' , 'NMF t-SNE', 'PCA UMAP', 'PCA t-SNE']
#ax.legend(legend_name, handletextpad=0.1, ncol=2, loc = 'upper right', framealpha = 0)

plt.savefig('../figures/mean_ari.png', bbox_inches='tight', dpi = 500)