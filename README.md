# CCP-scRNAseq-UMAP-TSNE

## Overview

Correlated Clustering and Projection (CCP) is a data-domain dimensionality reduction algorithm. In this repository, you will find the code used to reproduce "Analyzing scRNA-seq data by CCP-assisted UMAP and t-SNE", which is under revision.

CCP performs dimensionality reduction in 2 steps. 
1) Correlated Clustering: The data's features (ie genes) is partitioned according to their similarity. In this paper, we utilized k-means clustering, but any partitioning scheme can be utilized
2) Projection: Using the gene clusters, the samples are projected using Flexibility Ridigity Index to obtain a single descriptor. This is essentially the weighted sum over the accumulation of the gene-gene correlation over the cells.

For full details, please refer to https://arxiv.org/abs/2206.04189.


## Dependency
1) Python==3.8.2
2) pandas==1.30
2) numpy==1.20.3
3) umap-lean==0.5.1
4) scikit-learn==0.24.2
5) matplotlib==3.4.2
6) wget==3.2
wget will be required for preprocessing the data. 

Please run the following in your new virtual environment.
' pip install requirement.txt '

## Running the code
There should not be any need to do modification to the code as long as your directory is set to the location of main.py

To run the code, use the following:
' python main.py dataset_name '

Currently, you can run GSE57249, GSE67835, GSE75748time, GSE75748cell, GSE82187, GSE84133human4, GSE84133mouse1, GSE94820

## Content of this repository
1) ./algorithm/ : contains the code for dimensionality reduction and clustering
2) ./features/ : location of the output features after running main.py
3) ./results/: contain the results for the benchmaerk
4) ./plot_code/ contain the codes to generate the plots
5) ./data/: the location of the data
6) main.py: used to run the experiments
7) SingleCellDataProcessing/: contains the processing code.

## Citation
1) Hozumi, Y., & Wei, G. W. (2023). Analyzing scRNA-seq data by CCP-assisted UMAP and t-SNE. arXiv preprint arXiv:2306.13750.
2) Hozumi, Y., Tanemura, K. A., & Wei, G. W. (2023). Preprocessing of Single Cell RNA Sequencing Data Using Correlated Clustering and Projection. Journal of Chemical Information and Modeling.
3) Hozumi, Y., Wang, R., & Wei, G. W. (2022). CCP: correlated clustering and projection for dimensionality reduction. arXiv preprint arXiv:2206.04189.


## Further inquiry.
Please Email hozumiyu@msu.edu for further questions

Last updated 09/04/2023.

