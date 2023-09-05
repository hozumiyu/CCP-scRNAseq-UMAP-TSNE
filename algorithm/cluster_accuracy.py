# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 21:58:25 2023

@author: yutah
"""

from sklearn.cluster import KMeans
import os, warnings
import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
'''
This file computes the kmeans clutering and the ARI and NMI for the result
'''


def computeKMeans(X, y, max_state = 30):
    X = StandardScaler().fit_transform(X)
    n_clusters = np.unique(y).shape[0]
    LABELS = np.zeros([max_state, X.shape[0]])
    for random_state in range(max_state):
        myKM = KMeans(n_clusters = n_clusters, random_state = 1).fit(X)
        LABELS[random_state, :] = myKM.labels_
    return LABELS


def computeARI(LABELS, y):
    ARI = np.zeros(LABELS.shape[0])
    for random_state in range(LABELS.shape[0]):
        ARI[random_state] = adjusted_rand_score(y, LABELS[0, :])
    return np.mean(ARI)


def computeNMI(LABELS, y):
    NMI = np.zeros(LABELS.shape[0])
    for random_state in range(LABELS.shape[0]):
        NMI[random_state] = normalized_mutual_info_score(y, LABELS[0, :])
    return np.mean(NMI)


def computeClusteringScore(X, y, max_state = 10):
    with warnings.catch_warnings():
        LABELS = computeKMeans(X, y, max_state)
    ari = computeARI(LABELS, y)
    nmi = computeNMI(LABELS, y)
    return ari, nmi