import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.stats.multitest as smm
from sklearn.metrics import confusion_matrix
from utils import get_DELN_lfcs
import matplotlib.pyplot as plt


def scanpy_sig_test(X, Y, method='t-test'):
    # Create Scanpy AnnData Object
    X_group = np.repeat("X", nx)  # Labels for group X
    Y_group = np.repeat("Y", ny)  # Labels for group Y

    adata = sc.AnnData(np.vstack([X, Y]))  # Combine X and Y
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]  # Gene names
    adata.obs["group"] = np.concatenate([X_group, Y_group])  # Assign group labels

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Run Differential Expression Analysis using Scanpy's t-test
    sc.tl.rank_genes_groups(adata, groupby="group", method=method, reference="X")

    # Extract DE Results
    de_results = pd.DataFrame({
        "gene": adata.var_names,
        "log2_fc": adata.uns["rank_genes_groups"]["logfoldchanges"]["Y"],
        "p_value": adata.uns["rank_genes_groups"]["pvals"]["Y"],
        "p_adj": adata.uns["rank_genes_groups"]["pvals_adj"]["Y"]
    })

    return de_results["log2_fc"], de_results["p_value"]


def get_test_results(adj_p_vals, true_lfcs):
    gt_sig_idx = (true_lfcs != 0).reshape((-1,))
    pred_sig_idx = (adj_p_vals < 0.05)
    conf_mat = confusion_matrix(gt_sig_idx, pred_sig_idx)
    tn, fp, fn, tp = conf_mat.ravel()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0  # True Positive Rate (Recall)
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0  # True Negative Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Positive Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False Negative Rate
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    # Print results
    print(f"TPR: {tpr:.2f}")
    print(f"TNR: {tnr:.2f}")
    print(f"FPR: {fpr:.2f}")
    print(f"FNR: {fnr:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print()


np.random.seed(0)
nx = 1000
ny = 1000
n_genes = 1500
mu1 = 100 + np.random.normal(0, 10, (1, n_genes)) * np.random.binomial(1, 0.1, (1, n_genes))
mu2 = 100 + np.random.normal(0, 20, (1, n_genes)) * np.random.binomial(1, 0.1, (1, n_genes))
d1 = 10
d2 = 1

r1 = 1 / d1
r2 = 1 / d2
p1 = 1 / (1 + d1 * mu1)
p2 = 1 / (1 + d2 * mu2)

# Generate synthetic gene expression data
X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

true_lfcs = np.log2(mu2 / mu1)

# scanpy in raw space
sc_lfcs, sc_p_vals = scanpy_sig_test(X, Y)
sc_adj_pvals = smm.multipletests(sc_p_vals, alpha=0.05, method='fdr_bh')[1]
plt.scatter(sc_lfcs, -np.log10(sc_adj_pvals))
plt.scatter(true_lfcs, -np.log10(sc_adj_pvals))
plt.axhline(-np.log10(0.05), -1000, 1000, color='red', label='adj $p$ < 0.05')
plt.ylabel("$-\log_{10}(p)$")
plt.xlabel('Estimated LFC')
plt.legend()
plt.show()
print("Scanpy Test Results: ")
get_test_results(sc_adj_pvals, true_lfcs)


DELN_lfcs, DELN_p_vals = get_DELN_lfcs(X, Y, test='z')
DELN_adj_pvals = smm.multipletests(DELN_p_vals, alpha=0.05, method='fdr_bh')[1]
plt.axhline(-np.log10(0.05), -1000, 1000, color='red', label='adj $p$ < 0.05')
plt.scatter(DELN_lfcs, -np.log10(DELN_adj_pvals), label='DELN LFC vs DELN adj $p$')
plt.scatter(true_lfcs, -np.log10(DELN_adj_pvals), label='True LFC vs DELN adj $p$')
plt.ylabel("$-\log_{10}(p)$")
plt.legend()
plt.xlabel('Estimated LFC')
plt.show()

print("DELN Test Results: ")
get_test_results(DELN_adj_pvals, true_lfcs)
