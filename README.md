# CCP-scRNAseq-UMAP-TSNE

**OVERVIEW**

Code for "Analyzing scRNA-seq data by CCP-assisted UMAP and t-SNE" currently under review.

To successfully reproduce the results, please modify the main.py. All the parameters have been defined, so uncomment the data of interest.

A toy example has been provided using GSE75748 cell data.

**Python Version**
1) pandas v1.30
2) numpy v1.20.3
3) umap-lean v0.5.1
4) scikit-learn v0.24.2
5) matplotlib v.3.4.2

**Content of this repository**
1) ./algorithm/ : contains the code for dimensionality reduction and clustering
2) ./plotting_features/ : contains the features to generate the figures
3) ./results/: contain the results for the benchmaerk
4) ./plots/ contain the codes to generate the plots
5) ./data/: please place the data into this folder
6) DR_CCP_TSNE_main.py contains the CCP-assisted t-SNE benchmark
7) DR_CCP_UMAP_main.py contains the CCP-assisted UMAP benchmark
8) DR_UMAP_TSNE_main.py contains the benchmark for unassisted UMAP and t-SNE
9) main.py the overall code. Please comment out the data you want to test, and assign the random seed

