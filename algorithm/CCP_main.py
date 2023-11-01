# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 20:48:53 2023

@author: yutah
"""

import os, warnings
from algorithm.CCP.ccp import CCP   #ccp file
import numpy as np
from sklearn.cluster import KMeans
'''
This file compute CCP reduction with the Low variance gene approach

After importing and preprocessing the data, a partition is computed for the genes
Then, CCP dimensionality reduction is computed
The file will be saved in './feature/nameofdata/'
'''


def divide_features(X, n_components, cutoff = 0.8, random_state = 1):
    '''
        This method construct the feature partition
        Input:
            X: numpy array, ROWS should be the SAMPLES, COLUMN should be the genes.
            n_components: the number of partition (n-1 parition will be based on kmeans)
            cutoff: the ratio of low variance genes that will be placed in the LV-gene
            random_state: state of kmeans
    
        Return:
            index_feature: type list
                index_feature[i] will be the i-th partition, and index_feature[i] is a list
                index_feature[i-1] will contain the gene index of the low-variance
    '''
    
    variance = np.var(X, axis = 0)
    non_zero_variance_index = np.where(variance > 1e-6)[0]  #get the index of nonzero variance
    
    
    numNonZero = non_zero_variance_index.shape[0]    #number of nonzero variance index
    
    numCutoff = int( np.ceil(numNonZero * cutoff) )    #number of features used
    arg_sort_variance = np.argsort(variance[non_zero_variance_index])[::-1] ##rank from highest to lowers, this is the index
    used_arg_sort_variance = arg_sort_variance[:numCutoff]    #get the index of the best variance
    used_index = non_zero_variance_index[used_arg_sort_variance]   #get the index of the best variance, remapped to original features
    used_index.sort()
    
    not_used_variance = [ np.where(variance <= 1e-6)[0], arg_sort_variance[numCutoff:] ]
    not_used_variance = np.concatenate(not_used_variance)
    not_used_variance.sort()
    with warnings.catch_warnings():
        myKM = KMeans(n_clusters = n_components-1, random_state = random_state, n_init = 10)
    myKM.fit( X[:, used_index].T)
    labels = myKM.labels_
    index_features = [[] for i in range(n_components)]
    for l in range(n_components - 1):
        current_label_index_temp = np.where(labels == l)[0]
        index_features[l] = used_index[current_label_index_temp]
    index_features[-1] = not_used_variance
    return index_features

def CCP_wrapper(data, X, n_components, cutoff, random_state = 1):
    # This is a wrapper file for LV-gene integrated CCP
    # X: data matrix
    # n_components: number of super-genes
    # cutoff: ratio of genes in the LV-genes
    # random_state: seed
    index_feature = divide_features(X = X, n_components = n_components, cutoff = cutoff,  random_state = random_state)
    myCCP = CCP(n_components = n_components)
    X_ccp = myCCP.fit_transform(X, index_feature)
    return X_ccp
    
