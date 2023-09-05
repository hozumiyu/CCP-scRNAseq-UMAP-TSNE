# -*- coding: utf-8 -*-
"""
Created on Sun May 14 13:41:21 2023

@author: Yuta
"""

import sys, os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#construct the number of genes in the low-variance genes for a given cutoff

data_path = '../data/'
data_process_path = '../SingleCellDataProcess/'
def drop_sample(X, y):
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
    return X[good_index, :], y[good_index]

def preprocess_data(X, y):
    X = np.log10(1+X).T
    X, y = drop_sample(X, y)
    
    
    return X, y

def load_X(data, data_path = data_path, data_process_path = data_process_path):
    inpath = data_path + '%s/'%(data)
    if not os.path.exists(inpath + '%s_labels.csv'%(data)):
        os.system('python %s/main.py '%(data_process_path) + data + ' ' + data_path + ' ' + data_process_path)
    X = pd.read_csv(inpath + '%s_data.csv'%(data))
    X = X.values[:, 1:].astype(float)
    return X


def load_y(data, data_path = data_path, data_process_path = data_process_path):
    
    inpath = data_path + '%s/'%(data)
    if not os.path.exists(inpath + '%s_labels.csv'%(data)):
        os.system('python %s/main.py '%(data_process_path) + data + ' ' + data_path + ' ' + data_process_path)
    y = pd.read_csv(inpath + '%s_labels.csv'%(data))
    y = np.array(list(y['Label'])).astype(int)
    return y


data = 'GSE75748time'
X = load_X(data); y = load_y(data)
X, y = preprocess_data(X, y)

variance = np.var(X, axis = 0)
variance.sort()
variance = variance[::-1]

numNonZero = np.where(variance > 1e-6)[0].shape[0]
max_var = np.max(variance)

fig = plt.figure(figsize = (4.5,3.5))
ax = fig.add_subplot()
ax.plot(np.arange(variance.shape[0]), variance, color = 'black', linewidth = 2)

color_vec = ['r', 'b', 'g', 'orange']
for idx, cutoff in enumerate([0.6, 0.7, 0.8, 0.9]):
    temp = numNonZero * cutoff
    
    number =int( variance.shape[0] - temp)
    ax.text(x = temp - 600, y = 0.63,s = '%d LV-genes'%number, color = color_vec[idx], rotation = 90)
    ax.vlines(temp, ymin = 0,  ymax = max_var,  ls=':', color = color_vec[idx])

legend_text = ['Variance', '$v_c=0.6$','$v_c=0.7$','$v_c=0.8$','$v_c=0.9$']

ax.set_xlim([0,variance.shape[0]])
ax.set_ylim([0, max_var])
ax.legend(legend_text, loc = 'upper left')
size = 10
plt.rc('font', size=size)          # controls default text sizes
plt.rc('axes', titlesize=size)     # fontsize of the axes title
plt.rc('axes', labelsize=size)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=size)    # fontsize of the tick labels
plt.rc('ytick', labelsize=size)    # fontsize of the tick labels
plt.rc('legend', fontsize=size)    # legend fontsize
plt.rc('figure', titlesize=size)  # fontsize of the figure title
#fig.savefig(outpath + 'results2_new.jpg', bbox_inches='tight', dpi = 500)
fig.savefig('../figures/%s_variance.png'%(data), bbox_inches='tight', dpi = 500)