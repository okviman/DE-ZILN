# LN's $t$-test
Differential expression analysis with LN's $t$-test

# Small test
RECOMB requests a small test for the reviewers. To this end, we included small_test.py in our repo. It reproduces a single run of the averaged results in Table 2 in the submission. To this end, please keep in mind that the results are not expected to exactly reproduce the results in our submission. To reproduce the table, instead follow the intructions below.

# Reproducing RECOMB submission results

### Section 4.1
To reproduce Fig. 1, run variance_vs_metric.py.

To reproduce the Table 2, first run large_scale_NB_DE_test.py (the NB parameters need to be adjusted according to the text in the submission in order to reproduce Table 1), and then run large_scale_NB_latex_tables.py with the correct path to the results generated via execution of the former script.


## Visium HD tests
    - Glomerular capsules in the  kidney sample
    The files are in notebooks/test folder
        - preprocess_VisumHD_Kidney.ipynb is the preprocessing step that creates two h5ad files
            - merged_blobs_in_cluster_5.h5ad: contains gene count matrix for individual capsules, use for UMI count sub-sampling tests
            - podocytes_2um.h5ad : contains gene count matrix for each 2um spot in each capsule, used for spot sub-sampling tests
        - vishd_test_parallel.py and vishd_test_de_parallel.py run the count sub-sampling tests, and plot_de_results.py and plot-fpr_results.py are used to create the plots
        - vishd_test_shape_split_shared.py run the spot sub-sampling tests and visualize_results.py is used to create the plots.  

