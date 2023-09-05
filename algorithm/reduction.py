# -*- coding: utf-8 -*-
"""
Created on Wed May  3 22:38:06 2023

@author: yutah
"""
import os
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.decomposition import NMF, PCA

def computeTSNE(X, perplexity, random_state):
    myTSNE = TSNE(random_state = random_state, perplexity = perplexity, init = 'pca')
    X_tsne = myTSNE.fit_transform(X)
    return X_tsne


def computeUMAP(X, n_neighbors, random_state):
    myUMAP = UMAP(random_state = random_state, n_neighbors = n_neighbors)
    X_umap = myUMAP.fit_transform(X)
    return X_umap
    
    
    
def computePCA(X, n_components, random_state):
    myPCA = PCA(n_components = n_components, random_state = random_state)
    X_pca = myPCA.fit_transform(X)
    return X_pca


def computeNMF(X, n_components, random_state):
    myNMF = NMF(n_components = n_components, random_state =random_state , init = 'nndsvda')
    X_nmf = myNMF.fit_transform(X)
    return X_nmf


def reduction_wrapper(X, method, outfile, random_state = 1, secondary_param = None, normalize = False, n_components = None):
    '''
        Wrapper file for the UMAP, tSNE, NMF and PCA dimensionality reduction
        Input:
            X: numpy array, ROWS should be the SAMPLES, COLUMN should be the genes
            method: 'NMF', 'PCA', 'UMAP' or 'TSNE'
            secondary_param: used for UMAP or TSNE. Either perplexity or n_neighbor value
            n_components: used for TSNE or PCA
    '''
    if method == 'UMAP':
        if normalize:
            X = StandardScaler().fit_transform(X)
        X_dr = computeUMAP(X, n_neighbors = secondary_param, random_state=random_state)
        np.save(outfile, X_dr)
    elif method == 'TSNE':
        if normalize:
            X = StandardScaler().fit_transform(X)
        X_dr = computeTSNE(X, perplexity = secondary_param, random_state = random_state)
        np.save(outfile, X_dr)
        
    if not os.path.exists(outfile):
        if method == 'UMAP':
            if normalize:
                X = StandardScaler().fit_transform(X)
            X_dr = computeUMAP(X, n_neighbors = secondary_param, random_state=random_state)
            np.save(outfile, X_dr)
        
        elif method == 'TSNE':
            if normalize:
                X = StandardScaler().fit_transform(X)
            X_dr = computeTSNE(X, perplexity = secondary_param, random_state = random_state)
            np.save(outfile, X_dr)
            
        elif method == 'PCA':
            X_dr = computePCA(X, n_components, random_state = random_state)
            np.save(outfile, X_dr)
        
        elif method == 'NMF':
            X_dr = computeNMF(X, n_components, random_state = random_state)
            np.save(outfile, X_dr)
        
    else:
        X_dr = np.load(outfile)
            
    
    return X_dr