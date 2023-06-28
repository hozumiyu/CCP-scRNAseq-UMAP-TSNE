# -*- coding: utf-8 -*-
"""
Created on Wed May  3 22:38:06 2023

@author: yutah
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
from algorithm.ccp_predivided import CCP
from sklearn.manifold import TSNE
from umap import UMAP


def computeCCP(X, n_components, index_features):
    myCCP = CCP(n_components = n_components, random_state = 1)
    X_ccp = myCCP.fit_transform(X, index_features)
    return X_ccp




def computeTSNE(X, perplexity):
    myTSNE = TSNE(random_state = 1, perplexity = perplexity, init = 'pca')
    X_tsne = myTSNE.fit_transform(X)
    return X_tsne


def computeUMAP(X, n_neighbors):
    myUMAP = UMAP(random_state = 1, n_neighbors = n_neighbors)
    X_umap = myUMAP.fit_transform(X)
    return X_umap
    
    
    
    