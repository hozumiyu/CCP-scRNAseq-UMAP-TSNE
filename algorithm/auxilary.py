# -*- coding: utf-8 -*-
"""
Created on Thu Dec 15 20:46:18 2022

@author: yutah
"""

import numpy as np
import pandas as pd
import csv
import os
from sklearn.cluster import KMeans

def makeFolder(outpath):
    try:
        os.makedirs(outpath)
    except:
        return
    return

def load_X(data, inpath, data_process_path):
    # load the data
    # data: the name of the data
    # inpath: the default is ./data/
    # data_process_path: the location of where SingleCellProcess is located.
    if not os.path.exists(inpath + data + '/'  +'%s_data.csv'%(data)):
        os.system('python %s/main.py '%(data_process_path) + data + ' ' + inpath + ' ' + data_process_path)
    inpath = inpath + data + '/'
    X = pd.read_csv(inpath + '%s_data.csv'%(data))
    X = X.values[:, 1:].astype(float)
    return X




def load_y(data, inpath, data_process_path):
    # load the labels
    # data: the name of the data
    # inpath: the default is ./data/
    # data_process_path: the location of where SingleCellProcess is located.
    if not os.path.exists(inpath + data + '/' + '%s_labels.csv'%(data)):
        os.system('python %s/main.py '%(data_process_path) + data + ' ' + inpath + ' ' + data_process_path)
    inpath = inpath + data + '/'
    y = pd.read_csv(inpath + '%s_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    return y



def drop_sample(X, y, min_cell = 15):
    # Used to drop cell types with less than some number
    # X, y: data and label
    # min_cell: minimum number of cells
    original = X.shape[0]
    labels = np.unique(y)
    good_index = []
    for l in labels:
        index = np.where(y == l)[0]
        if index.shape[0] > 15:
            good_index.append(index)
        else:
            print('label %d removed'%(l))
    good_index = np.concatenate(good_index)
    good_index.sort()
    new = good_index.shape[0]
    print(original - new, 'samples removed')
    return X[good_index, :], y[good_index]


def preprocess_data(X, y, min_cell =15):
    # Preprocessing the data, and drop cell types with fewer than min_cell
    X = np.log10(1+X).T
    X, y = drop_sample(X, y, min_cell)
    
    
    return X, y
