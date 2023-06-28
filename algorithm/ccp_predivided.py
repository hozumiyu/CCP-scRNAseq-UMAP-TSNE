# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 13:52:12 2021

@author: yutah
"""

import numpy as np
from sklearn.metrics import pairwise_distances
import time
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
class CCP():
    def __init__(self, scale = 6, power=2, ktype = "exp", metric = 'euclidean', cutoff = None, n_components = 20, random_state = 1):
        self.scale = scale
        self.power = power
        self.ktype = ktype 
        self.metric = metric
        self.n_components = n_components
        self.seed = random_state
        self.avgmindist = np.zeros([self.n_components])   #initialize the scaling for each component
        if cutoff  == None:
            self.cutoff = np.zeros([self.n_components])
        else:
            self.cutoff = np.zeros([self.n_components]) * cutoff
        
        np.random.seed(1)  #random seeding for reproductivity
    def exponentialKernel(self, X, scale, cutoff):
        X_temp = np.power(X/scale, self.power)
        X_temp = np.exp(-X_temp)
        X_temp[X > cutoff] = 0
        #X_temp[X <1e-8] = 0
        return X_temp

    def lorentzkernel(self,X, scale, cutoff):
        X_temp = np.power(X/scale, self.power)
        X_temp = 1/ (1 + X_temp)
        X_temp[X > cutoff] = 0
        #X_temp[X <1e-8] = 0
        return X_temp

    def computeDensity(self, X, scale, cutoff):
        if self.ktype == "exp":
            return self.exponentialKernel(X, scale, cutoff)
        elif self.ktype == "lor":
            return self.lorentzkernel(X, scale, cutoff)
        else:
            raise ValueError("ktype must be exp or lor")
    
    def check_matrix(self, X):
        try:
            if X == None:
                return False
            else:
                return True
        except:
            if X.any():
                return True
            else:
                return False
        print("error")
        
            
    def computeCorrelation(self, index_component, index_feature, X = None, transform = False):
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
        correlation = self.computeDensity(correlation, scale_temp, self.cutoff[index_component])  #compute the density map

        return correlation
            
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
    
    def divideFeature(self):
        '''
            Divide the feature vector into numComponent
        '''
        
        index = []
        index2 = []
        for idx in range(self.numFeature):
            if np.var(self.X[:, idx]) > 1e-8:
                index.append(idx)
            else:
                index2.append(idx)
                
        X = self.X[:, index].copy()
        
        kmedoids = KMeans(n_clusters = self.n_components, max_iter = int(1e9), random_state = self.seed).fit(X.T)
        labels = kmedoids.labels_
        index_feature = [ [] for i in range(self.n_components)]
        
        for idx in range(labels.shape[0]):
            index_feature[ labels[idx] ].append(index[idx])
        
        return index_feature
    
    def fit(self, X, index_features):
        '''
            Fit the space with respect to X
            Parameters:
                X: the data. np.array of size [numSample, numFeature]
                index_feature = [[start1, end1], ..., [start_n, end_n]] . Default is none. Make sure that the list is int type
                    How the features should be divided. This does allow overlaps, though not tested in the paper
                    n should match the numComponent
                subSample: float less than 1. Default is none
                    Number of data points to sample for embedding
        '''
        #X is the manifold we will embedd our data in
        self.X = X
        self.numSample, self.numFeature = X.shape  #get the dimension
        print("Fitting Dataset. Datset size", X.shape)
        #if subSample is defined, shuffle data and obtain the subsampled X

        self.n_components = len(index_features)
        self.index_feature = index_features
        
        
        
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
            #print(index_component)
            correlation  = self.computeCorrelation(index_component, index_feature, X = X, transform = True)
            correlation = np.sum(correlation, axis = 1)
            Feature[:, index_component] = correlation
        return Feature
        
    
    def fit_transform(self, X, index_features):
        print("No subsampling. Embedding and transforming at the same time")
        self.X = X
        self.numSample, self.numFeature = self.X.shape
        #self.correlation = [[] for i in range(self.numFeature)]
        #self.weights = [[] for i in range(self.numFeature)]
        self.n_components = len(index_features)
        self.index_feature = index_features
        Feature = np.zeros([self.numSample, self.n_components])
        for index_component, index_feature in enumerate(self.index_feature):
            correlation  = self.computeCorrelation(index_component, index_feature)
            correlation = np.mean(correlation, axis = 1)
            Feature[:, index_component] = correlation
        return Feature
        
    
    