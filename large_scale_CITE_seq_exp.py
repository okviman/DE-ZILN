import anndata as ann
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import statsmodels.stats.multitest as smm
from scanpy_ttest import get_test_results, scanpy_sig_test
from tqdm import tqdm

def plot(ax, true_lfc, est_lfc, title, xlims, ylims, ylabel=False):
    ax.scatter(true_lfc, est_lfc)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    global_min = np.min([xlims[0], ylims[0]])
    global_max = np.max([xlims[1], ylims[1]])
    ax.plot([global_min, global_max], 
            [global_min, global_max], 'r--', label='Identity Line (y=x)')
    ax.set_xlabel("True Log2 Fold Change", fontsize=12)
    if ylabel:
        ax.set_ylabel("Estimated Log2 Fold Change (lfc)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')


# Load the base data
memory_CD4 = ann.read_h5ad("simul/10X_PBMC_10K/memory_CD4.h5ad")

np.random.seed(1)
replicates = 100
min_cells_per_group = 3
methods = ["deln", "scanpy"]

# Lists to store results from each replicate
mse_deln_list = []
mse_scanpy_list = []
all_results_list = []
metrics_list = []

for i in tqdm(range(replicates), desc="Running replicates"):

    # Create a copy to prevent data contamination
    adata_rep = memory_CD4.copy()

    # 1. Assign random groups
    group_assignments = np.random.choice(['group_A', 'group_B'], size=adata_rep.n_obs)
    adata_rep.obs['group'] = pd.Categorical(group_assignments) 

    # 2. Define signal genes and true LFCs
    target_group = 'group_B'
    n_genes = adata_rep.n_vars
    n_signal_genes = 100
    log2_fold_change = np.random.normal(loc=0.0, scale=1.0, size=n_signal_genes)
    fold_change = np.power(2, log2_fold_change)
    gene_indices_to_modify = np.random.choice(n_genes, size=n_signal_genes, replace=False)
    signal_gene_names = adata_rep.var_names[gene_indices_to_modify]

    adata_rep.var['is_signal_gene'] = False
    adata_rep.var.loc[signal_gene_names, 'is_signal_gene'] = True
    genes_to_modify_mask = adata_rep.var['is_signal_gene']

    true_lfcs = np.zeros(n_genes)
    true_lfcs[genes_to_modify_mask] = log2_fold_change

    # 3. Apply signal (modify raw counts)
    cells_to_modify_mask = adata_rep.obs['group'] == target_group
    adata_dense = adata_rep.X.toarray()

    # Y_data is the original data for group_A (unperturbed) 
    Y_data = adata_dense[~cells_to_modify_mask, :]
    
    # X_data_orig is the original data for group_B
    X_data_orig = adata_dense[cells_to_modify_mask, :]
    
    # Create the modified X data (apply perturbation)
    X_data_modified = X_data_orig.copy()
    X_data_modified[:, genes_to_modify_mask] *= fold_change

    expr_in_A = np.sum(Y_data > 0, axis=0) >= min_cells_per_group
    expr_in_B = np.sum(X_data_orig > 0, axis=0) >= min_cells_per_group
    gene_filter_mask = expr_in_A & expr_in_B

    X_data_filtered = X_data_modified[:, gene_filter_mask]
    Y_data_filtered = Y_data[:, gene_filter_mask]
    true_lfcs_filtered = true_lfcs[gene_filter_mask]
    signal_mask_filtered = genes_to_modify_mask[gene_filter_mask]
    filtered_gene_names = adata_rep.var_names[gene_filter_mask]

    # 4. Run DE tests on the raw count arrays
    
    # Method 1: DELN
    lfcs_deln, pvals_deln, se_deln = get_DELN_lfcs(X_data_filtered, Y_data_filtered, return_standard_error=True)
    adj_pvals_deln = smm.multipletests(pvals_deln, alpha=0.05, method='fdr_bh')[1]

    # Method 2: Wilcoxon
    corr_method="benjamini-hochberg"
    lfcs_w, adj_pvals_w = scanpy_sig_test(X_data_filtered, Y_data_filtered, method="wilcoxon", corr_method=corr_method)
    lfcs_scanpy = -lfcs_w.to_numpy()

    # Method 3: t-test 
    lfcs_t, adj_pvals_t = scanpy_sig_test(X_data_filtered, Y_data_filtered, method="t-test", corr_method=corr_method)

    # 5. Store results    
    de_results = pd.DataFrame({
        'ln_lfc': lfcs_deln, 
        'scanpy_lfc': lfcs_scanpy,
        'true_lfc': true_lfcs_filtered,
        'ln_adj_pvalue':  adj_pvals_deln,
        'w_adj_pvalue': adj_pvals_w.values,
        't_adj_pvalue': adj_pvals_t.values,
        'is_signal_gene': signal_mask_filtered
    }, index=adata_rep.var_names[gene_filter_mask])
    
    de_results['replicate'] = i
    all_results_list.append(de_results)

    # 6. Calculate and store MSE for this replicate
    mse_deln_list.append(np.mean((de_results["ln_lfc"] - de_results["true_lfc"])**2))
    mse_scanpy_list.append(np.mean((de_results["scanpy_lfc"] - de_results["true_lfc"])**2))

    # 7. Compute metrics.
    ln_results = get_test_results(adj_pvals_deln, true_lfcs_filtered, verbose=False)
    w_results = get_test_results(adj_pvals_w, true_lfcs_filtered, verbose=False)
    t_test_results = get_test_results(adj_pvals_t, true_lfcs_filtered, verbose=False)
    metrics = {"LN": ln_results, "Wilcoxon": w_results, "t-test": t_test_results}
    metrics_df = pd.DataFrame(metrics)
    metrics_df["replicate"] = i
    metrics_list.append(metrics_df)

    # 8. Generate a figure of estimated vs true fold change. 
    # Check these figures to make sure the runs were successful.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
    all_vals = np.concatenate([
        de_results["ln_lfc"],
        de_results["scanpy_lfc"]
    ])
    ylims = [np.min(all_vals) - 0.2, np.max(all_vals) + 0.2]
    xlims = [np.min(de_results["true_lfc"]) - 0.2, np.max(de_results["true_lfc"]) + 0.2]
    plot(ax1, de_results['true_lfc'], de_results['ln_lfc'], "LN vs. True LFC", xlims, ylims, ylabel=True)
    plot(ax2, de_results['true_lfc'], de_results['scanpy_lfc'], "Scanpy vs. True LFC", xlims, ylims, ylabel=False)
    fig.savefig(f"results/figures/lfc_{i}.png")
    plt.close(fig)

print("...Simulation finished.")

print("  Aggregate MSE Results")
print(f"DELN MSE:        {np.mean(mse_deln_list):.4f} +/- {np.std(mse_deln_list):.4f}")
print(f"Wilcoxon MSE:    {np.mean(mse_scanpy_list):.4f} +/- {np.std(mse_scanpy_list):.4f}")

# Combine all results into one big DataFrame
all_results_df = pd.concat(all_results_list)
all_results_df.to_csv("results/CITE_seq_results.csv")

all_metrics_df = pd.concat(metrics_list)
all_metrics_df.to_csv("results/CITE_seq_metrics.csv")
