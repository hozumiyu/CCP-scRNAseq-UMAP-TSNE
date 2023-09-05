# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:38:07 2023

@author: yutah
"""
import numpy as np

def feature_kmedoids(X, n_components, partition_metric = 'correlation', random_state = 1):
    from sklearn_extra.cluster import KMedoids
    from sklearn.metrics import pairwise_distances
    X = pairwise_distances(X.T, metric = partition_metric)
    myKMedoids = KMedoids(n_clusters = n_components, metric = 'precomputed', max_iter=int(1e9), random_state = random_state)
    myKMedoids.fit(X)
    labels = myKMedoids.labels_
    return labels


def feature_kmeans(X, n_components, random_state = 1):
    from sklearn.cluster import KMeans
    myKM = KMeans(n_clusters = n_components, random_state = random_state, max_iter = int(1e9))
    myKM.fit(X.T)
    labels = myKM.labels_
    return labels
    
    
def partition_features(X, n_components, partition_method, partition_metric, random_state):
    feature_variance = np.var(X, axis = 0)
    index = np.where(feature_variance> 1e-6)[0]
    bad_index =  np.where(feature_variance <= 1e-6)[0]
    print('Removing %d for low variance'%(len(bad_index)))
            
    X = X[:, index].copy()
    if partition_method == 'kmeans':
        labels = feature_kmeans(X, n_components, random_state = 1)
    elif partition_method == 'kmedoids':
        labels = feature_kmedoids(X, n_components, partition_metric = 'correlation', random_state = 1)
        
    index_feature = [ [] for i in range(n_components)]
    
    for idx in range(labels.shape[0]):
        index_feature[ labels[idx] ].append(index[idx])
        
    return index_feature, index, bad_index
    
