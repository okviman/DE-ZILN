# LN's $t$-test
Differential expression analysis with LN's $t$-test


# Visium HD tests
    - Glomerular capsules in the  kidney sample
    The files are in notebooks/test folder
        - preprocess_VisumHD_Kidney.ipynb is the preprocessing step that creates two h5ad files
            - merged_blobs_in_cluster_5.h5ad: contains gene count matrix for individual capsules, use for UMI count sub-sampling tests
            - podocytes_2um.h5ad : contains gene count matrix for each 2um spot in each capsule, used for spot sub-sampling tests
        - vishd_test_parallel.py and vishd_test_de_parallel.py run the count sub-sampling tests, and plot_de_results.py and plot-fpr_results.py are used to create the plots
        - vishd_test_shape_split_shared.py run the spot sub-sampling tests and visualize_results.py is used to create the plots.  

