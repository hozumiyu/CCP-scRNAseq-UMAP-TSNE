# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:52:12 2021

@author: yutah
"""

import numpy as np
from sklearn.metrics import pairwise_distances
from algorithm.CCP.feature_partition import partition_features
from algorithm.CCP.kernel import computeKernel


class CCP():
    def __init__(self, scale = 6, power=2, ktype = "exp", metric = 'euclidean', n_components = 20, 
                 partition_method = 'kmeans', partition_metric = 'correlation', random_state = 1):
        self.scale = scale
        self.power = power
        self.ktype = ktype 
        self.metric = metric
        self.n_components = n_components
        self.random_state = random_state
        self.avgmindist = np.zeros([self.n_components])   #initialize the scaling for each component
        self.cutoff = np.zeros([self.n_components])
        self.partition_method = partition_method
        self.partition_metric = partition_metric
        
    
    def divideFeature(self):
        '''
            Divide the feature vector into numComponent
        '''
        index_feature, index, bad_index = partition_features(self.X, n_components = self.n_components, partition_method = self.partition_method, 
                           partition_metric = self.partition_metric, random_state = self.random_state)
        
        print('removing %d features for low variance'%(len(bad_index)))
        self.bad_index = bad_index
        self.index = index
        return index_feature
            
    def compute_descriptor(self, index_component, index_feature, X = None, transform = False):
        #correlation = self.X[:, index_feature]
        if transform:
            correlation = pairwise_distances(X[:, index_feature], self.X[:, index_feature], metric = self.metric)
        else:
            correlation = pairwise_distances(self.X[:, index_feature], metric = self.metric)
            
        if self.avgmindist[index_component] == 0.:
            self.avgmindist[index_component] = self.computeAvgMinDistance(correlation)
            
        if self.cutoff[index_component] == 0.:
            #cutoffMatrix = np.reshape(correlation[correlation > 1e-8], -1)
            avg = np.mean(correlation )
            std = np.std(correlation )
            self.cutoff[index_component] = avg + 3*std
            
        scale_temp = self.scale * self.avgmindist[index_component]  #get the scale
        correlation = computeKernel(correlation, ktype = self.ktype, scale = scale_temp, power = self.power, cutoff = self.cutoff[index_component])

        descriptor = np.sum(correlation, axis = 1)
        return descriptor
            
    def computeAvgMinDistance(self, X):
        #Compute the average minimum distance for scaling in lorentz and exponential function
        minDistance = []
        for idx in range(X.shape[0]):  #loop to get the nonzero minimum distance for each data
            nonzero = X[idx,:]
            nonzero = nonzero[nonzero > 1e-8]
            if nonzero.shape[0] > 0:
                minDistance.append(np.min(nonzero))
        if len(minDistance) == 0:
            return 0
        else:
            avgmindist = np.mean(minDistance)  #get the average
        return avgmindist
    
    
    
    def fit(self, X, index_feature):
        '''
            Fit the space with respect to X
            Parameters:
                X: the data. np.array of size [numSample, numFeature]
        '''
        #X is the manifold we will embedd our data in
        self.X = X
        self.numSample, self.numFeature = X.shape  #get the dimension
        print("Fitting Dataset. Datset size", X.shape)
        #if subSample is defined, shuffle data and obtain the subsampled X
        
        if type(index_feature) == list:
            self.index_feature = index_feature
            self.n_components = len(index_feature)
            self.avgmindist = np.zeros([self.n_components])   #initialize the scaling for each component
            self.cutoff = np.zeros([self.n_components])
        else:
            self.index_feature = self.divideFeature()   #split the features into n-components
        print(self.n_components)
        
        #compute the avg minimum distance induced by X
        for idx_nc in range(self.n_components):
            dist_i = pairwise_distances(self.X[:, self.index_feature[idx_nc]])
            self.avgmindist[idx_nc] = self.computeAvgMinDistance(dist_i)
            avg = np.mean(dist_i)
            std = np.std(dist_i)
            self.cutoff[idx_nc] = avg + 3*std
        print("fitting complete")
        return
    
    def transform(self, X):
        print("Transforming data. Dimension of X:", X.shape, "Embedding Space's size:", self.X.shape)
        numSample = X.shape[0]
        Feature = np.zeros([numSample, self.n_components])
        for index_component, index_feature in enumerate(self.index_feature):
            descriptor  = self.compute_descriptor(index_component, index_feature, X = X, transform = True)
            Feature[:, index_component] = descriptor
        return Feature
        
    
    def fit_transform(self, X, index_feature = None):
        self.X = X
        self.numSample, self.numFeature = self.X.shape
        if type(index_feature) == list:
            self.index_feature = index_feature
            self.n_components = len(index_feature)
            self.avgmindist = np.zeros([self.n_components])   #initialize the scaling for each component
            self.cutoff = np.zeros([self.n_components])
        else:
            self.index_feature = self.divideFeature()   #split the features into n-components
        Feature = np.zeros([self.numSample, self.n_components])
        for index_component, index_feature in enumerate(self.index_feature):
            descriptor  = self.compute_descriptor(index_component, index_feature)
            Feature[:, index_component] = descriptor
        return Feature
        
    
    
