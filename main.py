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
data = sys.argv[1]
random_state = sys.argv[2]
#option = int(sys.argv[1])
#random_state = int(sys.argv[2])

data_process_path = './SingleCellDataProcess/'   #change this line if you have a different path for the data processing
data_path = './data/'  #location of where the processed raw data will be saved


#data, scaling, cutoff, n_components = 'GSE67835', 'std', 0.7,300
#data, scaling, cutoff, n_components = 'GSE75748cell', 'raw', 0.7,250
#data, scaling, cutoff, n_components = 'GSE75748time', 'std', 0.7,300
#data, scaling, cutoff, n_components = 'GSE82187', 'raw', 0.8,150
#data, scaling, cutoff, n_components = 'GSE84133human3', 'std', 0.5,200
#data, scaling, cutoff, n_components = 'GSE84133human4', 'raw', 0.6,150
#data, scaling, cutoff, n_components = 'GSE84133mouse1', 'raw', 0.8,250
#data, scaling, cutoff, n_components = 'GSE94820', 'std', 0.5,250
#data, scaling, cutoff, n_components = 'GSE57249', 'std', 0.7,50

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

outpath = './features/%s_features/'%(data)
makeFolder(outpath)

#Compute CCP and the CCP assisted UMAP and tSNE

X_ccp = CCP_wrapper(data, X, n_components, cutoff, random_state = random_state)
ari_ccp, nmi_ccp = computeClusteringScore(X_ccp, y, max_state = 10)
X_ccp = X_ccp/X_ccp.shape[0]

outfile = '%s_ccp_tsne_state%d.npy'%(data, random_state)
X_ccp_tsne = reduction_wrapper(X = X_ccp, method = 'TSNE', outfile = outpath + outfile, secondary_param = perplexity, normalize = scaling, random_state = random_state)
ari_ccp_tsne, nmi_ccp_tsne = computeClusteringScore(X_ccp_tsne, y, max_state = 10)


outfile = '%s_ccp_umap_state%d.npy'%(data, random_state)
X_ccp_umap = reduction_wrapper(X = X_ccp, method = 'UMAP', outfile = outpath + outfile, secondary_param = n_neighbors, normalize = scaling, random_state = random_state)
ari_ccp_umap, nmi_ccp_umap = computeClusteringScore(X_ccp_umap, y, max_state = 10)

if data == 'GSE57249':
    n_components = 10


#Compute NMF and NMF assisted UMAP and tSNE

outfile = '%s_nmf_state%d.npy'%(data, random_state)
X_nmf = reduction_wrapper(X, method = 'NMF', outfile = outpath + outfile, n_components = n_components, random_state = random_state)
ari_nmf, nmi_nmf = computeClusteringScore(X_nmf, y, max_state = 10)

outfile = '%s_nmf_tsne_state%d.npy'%(data, random_state)
X_nmf_tsne = reduction_wrapper(X = X_nmf, method = 'TSNE', outfile = outpath + outfile, secondary_param = perplexity, normalize = False, random_state = random_state)
ari_nmf_tsne, nmi_nmf_tsne = computeClusteringScore(X_nmf_tsne, y, max_state = 10)

outfile = '%s_nmf_umap_state%d.npy'%(data, random_state)
X_nmf_umap = reduction_wrapper(X = X_nmf, method = 'UMAP', outfile = outpath + outfile, secondary_param = n_neighbors, normalize = False, random_state = random_state)
ari_nmf_umap, nmi_nmf_umap = computeClusteringScore(X_nmf_umap, y, max_state = 10)


#Compute PCA and PCA assisted UMAP and tSNE

outfile = '%s_pca_state%d.npy'%(data, random_state)
X_pca = reduction_wrapper(X, method = 'PCA', outfile = outpath + outfile, n_components = n_components)
ari_pca, nmi_pca = computeClusteringScore(X_pca, y, max_state = 10)

outfile = '%s_pca_tsne_state%d.npy'%(data, random_state)
X_pca_tsne = reduction_wrapper(X = X_pca, method = 'TSNE', outfile = outpath + outfile, secondary_param = perplexity, normalize = True, random_state = random_state)
ari_pca_tsne, nmi_pca_tsne = computeClusteringScore(X_pca_tsne, y, max_state = 10)

outfile = '%s_pca_umap_state%d.npy'%(data, random_state)
X_pca_umap = reduction_wrapper(X = X_pca, method = 'UMAP', outfile = outpath + outfile, secondary_param = n_neighbors, normalize = True, random_state = random_state)
ari_pca_umap, nmi_pca_umap = computeClusteringScore(X_pca_umap, y, max_state = 10)



#Compute UMAP and tSNE
variance = np.var(X, axis = 0)
index = np.where(variance > 1e-6)[0]
X = X[:, index]
outfile = '%s_tsne_state%d.npy'%(data, random_state)
X_tsne = reduction_wrapper(X = X, method = 'TSNE', outfile = outpath + outfile, secondary_param = perplexity, normalize = False, random_state = random_state)
ari_tsne, nmi_tsne = computeClusteringScore(X_tsne, y, max_state = 10)

outfile = '%s_umap_state%d.npy'%(data, random_state)
X_umap = reduction_wrapper(X = X, method = 'UMAP', outfile = outpath + outfile, secondary_param = n_neighbors, normalize = False, random_state = random_state)
ari_umap, nmi_umap = computeClusteringScore(X_umap, y, max_state = 10)


outpath_results = './results/%s_results/'%(data); makeFolder(outpath_results)
file = open(outpath_results + '%s_results_state%d.csv'%(data, random_state), 'w')
file.write( 'Metric,CCP,CCP TSNE,CCP UMAP,NMF,NMF TSNE,NMF UMAP,PCA,PCA TSNE,PCA UMAP,TSNE,UMAP\n')
file.write( 'ARI,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'%(ari_ccp, ari_ccp_tsne, ari_ccp_umap, ari_nmf, ari_nmf_tsne, ari_nmf_umap, ari_pca, ari_pca_tsne, ari_pca_umap, ari_tsne, ari_umap))
file.write( 'NMI,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'%(nmi_ccp, nmi_ccp_tsne, nmi_ccp_umap, nmi_nmf, nmi_nmf_tsne, nmi_nmf_umap, nmi_pca, nmi_pca_tsne, nmi_pca_umap, nmi_tsne, nmi_umap))

if plot:
    fig, ax = plt.subplots(nrows = 2, ncols=4)
    ax[0, 0].scatter(X_ccp_tsne[:, 0], X_ccp_tsne[:, 1], c = y, s=  1)
    ax[1, 0].scatter(X_ccp_umap[:, 0], X_ccp_umap[:, 1], c = y, s = 1)
    
    ax[0, 1].scatter(X_pca_tsne[:, 0], X_pca_tsne[:, 1], c = y, s=  1)
    ax[1, 1].scatter(X_pca_umap[:, 0], X_pca_umap[:, 1], c = y, s=  1)
    
    ax[0, 2].scatter(X_nmf_tsne[:, 0], X_nmf_tsne[:, 1], c = y, s=  1)
    ax[1, 2].scatter(X_nmf_umap[:, 0], X_nmf_umap[:, 1], c = y, s=  1)
    
    ax[0, 3].scatter(X_tsne[:, 0], X_tsne[:, 1], c = y, s=  1)
    ax[1, 3].scatter(X_umap[:, 0], X_umap[:, 1], c = y, s=  1)
    
    
