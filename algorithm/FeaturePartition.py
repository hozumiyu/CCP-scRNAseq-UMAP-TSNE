# -*- coding: utf-8 -*-
"""
Created on Thu Jun 15 17:04:21 2023

@author: Yuta
"""
from sklearn.cluster import KMeans
import numpy as np


def writeIndex(outfile, index_features):
    file = open(outfile, 'w')
    for index in index_features:
        index = list(index)
        file.writelines(str(index) + '\n')
    file.close()
    return

def unpackIndex(infile):
    file = open(infile, 'r')
    lines = file.readlines()
    index_features = []
    for line in lines:
        line = eval(line)
        line = np.array(line)
        line = line.astype(int)
        index_features.append(line)
    return index_features

def divide_features(X, n_components, cutoff = 0.8, random_state = 1):
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
    
    print('Not used: %d'%not_used_variance.shape[0])
    
    myKM = KMeans(n_clusters = n_components-1, random_state = random_state, n_init = 10)
    myKM.fit( X[:, used_index].T)
    labels = myKM.labels_
    index_features = [[] for i in range(n_components)]
    for l in range(n_components - 1):
        current_label_index_temp = np.where(labels == l)[0]
        index_features[l] = used_index[current_label_index_temp]
    index_features[-1] = not_used_variance
    return index_features