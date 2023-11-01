# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:41:21 2023

@author: Yuta
"""

import sys, os
import numpy as np
from algorithm.auxilary import load_X, load_y, preprocess_data, makeFolder
from algorithm.CCP_main import CCP_wrapper
from algorithm.reduction import reduction_wrapper
import matplotlib.pyplot as plt
from algorithm.cluster_accuracy import computeClusteringScore




plot = True
data = 'GSE75748cell'
random_state = 1
#data = sys.argv[1]
#random_state = sys.argv[2]

data_process_path = './SingleCellDataProcess/'   #change this line if you have a different path for the data processing
data_path = './data/'  #location of where the processed raw data will be saved


#the parameters for each dataset to reproduce results
if data == 'GSE67835':
    scaling, cutoff, n_components = True, 0.7,300
elif data == 'GSE75748cell':
    scaling, cutoff, n_components = False, 0.7,250
elif data == 'GSE75748time':
    scaling, cutoff, n_components = True,  0.7, 300
elif data == 'GSE82187':
    scaling, cutoff, n_components = False, 0.8,150
elif data == 'GSE84133human4': 
    scaling, cutoff, n_components = False, 0.6,150
elif data == 'GSE84133mouse1':
    scaling, cutoff, n_components = False, 0.8,250
elif data == 'GSE94820':
    scaling, cutoff, n_components = True, 0.5,250
elif data == 'GSE57249': 
    scaling, cutoff, n_components = True, 0.7,50



X = load_X(data, data_path, data_process_path)
y = load_y(data, data_path, data_process_path)
if data != 'GSE57249':
    X, y = preprocess_data(X, y)
    perplexity = 30
    n_neighbors = 15
else:
    X = np.log10(1+X).T
    perplexity = 5
    n_neighbors = 3


print('Computing Dimensionality Reduction with the following parameters:')
print('data:', data, 'cutoff:', cutoff, 'n_components:', n_components)

#compute CCP
X_ccp = CCP_wrapper(data, X, n_components, cutoff, random_state = random_state)

#compute CCP assisted UMAP
X_ccp_umap = reduction_wrapper(X = X_ccp, method = 'UMAP', secondary_param = n_neighbors, normalize = scaling, random_state = random_state)
ari_ccp_umap, nmi_ccp_umap = computeClusteringScore(X_ccp_umap, y, max_state = 10)

print('CCP-UMAP. ARI:, ', ari_ccp_umap, 'NMI:', nmi_ccp_umap)


#compute CCP assisted TSNE
X_ccp_tsne = reduction_wrapper(X = X_ccp, method = 'TSNE', secondary_param = perplexity, normalize = scaling, random_state = random_state)
ari_ccp_tsne, nmi_ccp_tsne = computeClusteringScore(X_ccp_tsne, y, max_state = 10)

print('CCP-TSNE. ARI:, ', ari_ccp_tsne, 'NMI:', nmi_ccp_tsne)


if data == 'GSE57249':
    n_components = 10


#Compute NMF and NMF assisted UMAP and tSNE

X_nmf = reduction_wrapper(X, method = 'NMF', n_components = n_components, random_state = random_state)


#NMF assisted UMAP
X_nmf_umap = reduction_wrapper(X = X_nmf, method = 'UMAP', secondary_param = n_neighbors, normalize = False, random_state = random_state)
ari_nmf_umap, nmi_nmf_umap = computeClusteringScore(X_nmf_umap, y, max_state = 10)

print('NMF-UMAP. ARI:, ', ari_nmf_umap, 'NMI:', nmi_nmf_umap)

#NMF assisted TSNE
X_nmf_tsne = reduction_wrapper(X = X_nmf, method = 'TSNE', secondary_param = perplexity, normalize = False, random_state = random_state)
ari_nmf_tsne, nmi_nmf_tsne = computeClusteringScore(X_nmf_tsne, y, max_state = 10)
print('NMF-UMAP. ARI:, ', ari_nmf_tsne, 'NMI:', nmi_nmf_tsne)


#PCA
X_pca = reduction_wrapper(X, method = 'PCA', n_components = n_components)

#PCA assisted UMAP
X_pca_umap = reduction_wrapper(X = X_pca, method = 'UMAP', secondary_param = n_neighbors, normalize = True, random_state = random_state)
ari_pca_umap, nmi_pca_umap = computeClusteringScore(X_pca_umap, y, max_state = 10)
print('PCA-UMAP. ARI:, ', ari_pca_umap, 'NMI:', nmi_pca_umap)

#PCA assisted TSNE
X_pca_tsne = reduction_wrapper(X = X_pca, method = 'TSNE', secondary_param = perplexity, normalize = True, random_state = random_state)
ari_pca_tsne, nmi_pca_tsne = computeClusteringScore(X_pca_tsne, y, max_state = 10)
print('PCA-TSNE. ARI:, ', ari_pca_tsne, 'NMI:', nmi_pca_tsne)



#Compute UMAP
X_umap = reduction_wrapper(X = X, method = 'UMAP', secondary_param = n_neighbors, normalize = False, random_state = random_state)
ari_umap, nmi_umap = computeClusteringScore(X_umap, y, max_state = 10)
print('UMAP. ARI:, ', ari_umap, 'NMI:', nmi_umap)

#compute TSNE
X_tsne = reduction_wrapper(X = X, method = 'TSNE', secondary_param = perplexity, normalize = False, random_state = random_state)
ari_tsne, nmi_tsne = computeClusteringScore(X_tsne, y, max_state = 10)
print('TSNE. ARI:, ', ari_tsne, 'NMI:', nmi_tsne)




if plot:
    fig, ax = plt.subplots(nrows = 2, ncols=4, figsize = (8,4))
    
    ax[0, 0].scatter(X_ccp_umap[:, 0], X_ccp_umap[:, 1], c = y, s = 1)
    ax[1, 0].scatter(X_ccp_tsne[:, 0], X_ccp_tsne[:, 1], c = y, s=  1)
    
    ax[0, 1].scatter(X_pca_umap[:, 0], X_pca_umap[:, 1], c = y, s=  1)
    ax[1, 1].scatter(X_pca_tsne[:, 0], X_pca_tsne[:, 1], c = y, s=  1)
    
    ax[0, 2].scatter(X_nmf_umap[:, 0], X_nmf_umap[:, 1], c = y, s=  1)
    ax[1, 2].scatter(X_nmf_tsne[:, 0], X_nmf_tsne[:, 1], c = y, s=  1)
    
    ax[0, 3].scatter(X_umap[:, 0], X_umap[:, 1], c = y, s=  1)
    ax[1, 3].scatter(X_tsne[:, 0], X_tsne[:, 1], c = y, s=  1)
    
    ax[0,0].set_title('CCP-assisted')
    ax[0,1].set_title('PCA-assisted')
    ax[0,2].set_title('NMF-assisted')
    ax[0,3].set_title('Unassisted')
    ax[0,0].set_ylabel('UMAP', rotation=90)
    ax[1,0].set_ylabel('TSNE', rotation=90)
    plt.setp(ax, xticks=[], yticks=[])
    
