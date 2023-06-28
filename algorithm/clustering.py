#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:44:51 2023

@author: yutaho
"""

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from algorithm.auxilary import load_X, load_y, preprocess_data, makeFolder
from algorithm.FeaturePartition import unpackIndex
import os
import warnings
def writeCSV(outfile, lines):
    file = open(outfile, 'w')
    for line  in lines:
        file.writelines(line)
    file.close()
    
    return

def km(X, y,  max_state = 30):
    n_clusters = np.unique(y).shape[0]
    ari = np.zeros(max_state)
    for idx in range(max_state):
        myKM = KMeans(n_clusters = n_clusters, random_state = idx, n_init = 10)
        myKM.fit(X)
        labels = myKM.labels_
        ari[idx] = adjusted_rand_score(y, labels)
    return ari

def runKM( X, y):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ari = km(X, y)
    return ari
                    
        
    