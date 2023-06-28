# -*- coding: utf-8 -*-
"""
Created on Sun Apr 23 13:27:07 2023

@author: yutah
"""
import numpy as np


def exponentialKernel(X, scale, power, cutoff):
    X_temp = np.power(X/scale, power)
    X_temp = np.exp(-X_temp)
    X_temp[X > cutoff] = 0
    #X_temp[X <1e-8] = 0
    return X_temp

def lorentzKernel(X, scale, power, cutoff):
    X_temp = np.power(X/scale, power)
    X_temp = 1/ (1 + X_temp)
    X_temp[X > cutoff] = 0
    #X_temp[X <1e-8] = 0
    return X_temp

def computeKernel(X, ktype, scale, power, cutoff):
    if ktype == 'exp':
        correlation = exponentialKernel(X, scale, power, cutoff)
    elif ktype == 'lor':
        correlation = lorentzKernel(X, scale, power, cutoff)
    else:
        print('ktype must be exp or lor')
    return correlation