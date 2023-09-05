# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 10:34:57 2023

@author: yutah
"""
import numpy as np
import pandas as pd


for option in range(9):
    if option == 0:
        data, scaling, cutoff, n_components = 'GSE67835', True, 0.7,300
    elif option == 1:
        data, scaling, cutoff, n_components = 'GSE75748cell', False, 0.7,250
    elif option == 2:
        data, scaling, cutoff, n_components = 'GSE75748time', True, 0.7,300
    elif option == 3:
        data, scaling, cutoff, n_components = 'GSE82187', False, 0.8,150
    elif option == 4: 
        data, scaling, cutoff, n_components = 'GSE84133human3', True , 0.5,200
    elif option == 5:
        data, scaling, cutoff, n_components = 'GSE84133human4', False, 0.6,150
    elif option == 6:
        data, scaling, cutoff, n_components = 'GSE84133mouse1', False, 0.8,250
    elif option == 7:
        data, scaling, cutoff, n_components = 'GSE94820', True, 0.5,250
    elif option == 8: 
        data, scaling, cutoff, n_components = 'GSE57249', True, 0.7,50
    
    ccp_ari = []; ccp_nmi = []
    ccp_umap_ari = []; ccp_umap_nmi = []
    ccp_tsne_ari = []; ccp_tsne_nmi = [];
    
    nmf_ari = []; nmf_nmi = []
    nmf_umap_ari = []; nmf_umap_nmi =[]
    nmf_tsne_ari = []; nmf_tsne_nmi = []
    
    pca_ari = []; pca_nmi = []
    pca_umap_ari = []; pca_umap_nmi = []
    pca_tsne_ari =[] ; pca_tsne_nmi = []
    
    umap_ari = []; umap_nmi = []
    tsne_ari = []; tsne_nmi = []
    
    random_state_vec = np.arange(1,11) 
    for random_state in random_state_vec:
        inpath = './results/%s_results/'%(data)
        file = '%s_results_%d.csv'%(data, random_state)
        results = pd.read_csv(inpath + file)
        
        a,n = results['CCP']
        ccp_ari.append(a); ccp_nmi.append(n)
        
        a, n = results['CCP UMAP']
        ccp_umap_ari.append(a); ccp_umap_nmi.append(n)
        
        a, n = results['CCP TSNE']
        ccp_tsne_ari.append(a); ccp_tsne_nmi.append(n)
        
        a,n = results['NMF']
        nmf_ari.append(a); nmf_nmi.append(n)
        
        a, n = results['NMF UMAP']
        nmf_umap_ari.append(a); nmf_umap_nmi.append(n)
        
        a, n = results['NMF TSNE']
        nmf_tsne_ari.append(a); nmf_tsne_nmi.append(n)
        
        a,n = results['PCA']
        pca_ari.append(a); pca_nmi.append(n)
        
        a, n = results['PCA UMAP']
        pca_umap_ari.append(a); pca_umap_nmi.append(n)
        
        a, n = results['PCA TSNE']
        pca_tsne_ari.append(a); pca_tsne_nmi.append(n)
        
        a, n = results['UMAP']
        umap_ari.append(a); umap_nmi.append(n)
        
        a, n = results['TSNE']
        tsne_ari.append(a); tsne_nmi.append(n)
        
    outpath = './results/'
    outfile = '%s_results.csv'%(data)
    file = open(outpath + outfile, 'w')
    file.write( 'Metric,CCP,CCP TSNE,CCP UMAP,NMF,NMF TSNE,NMF UMAP,PCA,PCA TSNE,PCA UMAP,TSNE,UMAP\n')
    file.write( 'ARI,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'%( np.mean(ccp_ari), np.mean(ccp_tsne_ari), np.mean(ccp_umap_ari), np.mean(nmf_ari), np.mean(nmf_tsne_ari), np.mean(nmf_umap_ari), np.mean(pca_ari), np.mean(pca_tsne_ari), np.mean(pca_umap_ari), np.mean(tsne_ari), np.mean(umap_ari)))
    
    
    
    file.write( 'NMI,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n'%( np.mean(ccp_nmi), np.mean(ccp_tsne_nmi), np.mean(ccp_umap_nmi), np.mean(nmf_nmi), np.mean(nmf_tsne_nmi), np.mean(nmf_umap_nmi), np.mean(pca_nmi), np.mean(pca_tsne_nmi), np.mean(pca_umap_nmi), np.mean(tsne_nmi), np.mean(umap_nmi), ))
        
        