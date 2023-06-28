# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:41:21 2023

@author: Yuta
"""

import sys, os
import numpy as np
from algorithm.auxilary import load_X, load_y, preprocess_data, makeFolder
from algorithm.FeaturePartition import writeIndex, divide_features, unpackIndex
from algorithm.reduction import computeCCP


#data, scaling, cutoff, n_components = 'GSE67835', 'std', 0.7,300
data, scaling, cutoff, n_components = 'GSE75748cell', 'raw', 0.7,250
#data, scaling, cutoff, n_components = 'GSE75748time', 'std', 0.7,300
#data, scaling, cutoff, n_components = 'GSE82187', 'raw', 0.8,150
#data, scaling, cutoff, n_components = 'GSE84133human3', 'std', 0.5,200
#data, scaling, cutoff, n_components = 'GSE84133human4', 'raw', 0.6,150
#data, scaling, cutoff, n_components = 'GSE84133mouse1', 'raw', 0.8,250
#data, scaling, cutoff, n_components = 'GSE94820', 'std', 0.5,250
#data, scaling, cutoff, n_components = 'GSE57249', 'std', 0.7,50

random_state = 1


# compute CCP reduction
os.system('python DR_CCP_TSNE_main.py %s %d %.1f %d %s'%(data, n_components, cutoff, random_state, scaling))


#compute CCP assisted t-SNE and UMAP
os.system('python DR_CCP_UMAP_main.py %s %d %.1f %d %s'%(data, n_components, cutoff, random_state, scaling))


os.system('python DR_UMAP_TSNE_main.py %s %d'%(data, random_state))