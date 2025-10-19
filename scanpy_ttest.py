import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.stats.multitest as smm
from sklearn.metrics import confusion_matrix
from utils import get_DELN_lfcs
import matplotlib.pyplot as plt


def scanpy_sig_test(X, Y, method='t-test', normalization='CP10K', corr_method="bonferroni"):
    nx = X.shape[0]
    ny = Y.shape[0]
    n_genes = Y.shape[1]
    # Create Scanpy AnnData Object
    X_group = np.repeat("X", nx)  # Labels for group X
    Y_group = np.repeat("Y", ny)  # Labels for group Y

    if normalization == 'median-of-ratios':
        X = X.astype(float)
        Y = Y.astype(float)
        X_ = X.copy()
        Y_ = Y.copy()
        X_[X_ <= 0] = np.nan
        Y_[Y_ == 0] = np.nan

        denom_Y = np.exp(np.nanmean(np.log(Y_), 0))
        c_Y = np.nanmedian(Y_ / denom_Y, 1, keepdims=True)
        Y /= c_Y

        denom_X = np.exp(np.nanmean(np.log(X_), 0))
        c_X = np.nanmedian(X_ / denom_X, 1, keepdims=True)
        X /= c_X

        Y[Y == np.nan] = 0.
        X[X == np.nan] = 0.

    adata = sc.AnnData(np.vstack([X, Y]))  # Combine X and Y
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]  # Gene names
    adata.obs["group"] = np.concatenate([X_group, Y_group])  # Assign group labels

    adata.layers["counts"] = adata.X.copy()
    if normalization == 'CP10K':
        sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Run Differential Expression Analysis using Scanpy's t-test
    #corr_method = "bonferroni"  # use Bonferroni to match output of Seurat
    sc.tl.rank_genes_groups(adata, groupby="group", method=method, reference="X", corr_method=corr_method)

    # Extract DE Results
    de_results = pd.DataFrame({
        "gene": adata.uns['rank_genes_groups']['names']['Y'],
        "log2_fc": adata.uns["rank_genes_groups"]["logfoldchanges"]["Y"],
        "p_value": adata.uns["rank_genes_groups"]["pvals"]["Y"],
        "p_adj": adata.uns["rank_genes_groups"]["pvals_adj"]["Y"]
    })
    de_results["gene"] = [int(gene.replace("Gene", "")) for gene in de_results["gene"]]
    gene_idx_sorted = np.argsort(de_results["gene"])

    return de_results["log2_fc"][gene_idx_sorted], de_results["p_adj"][gene_idx_sorted]


def get_test_results(adj_p_vals, true_lfcs, verbose=True):
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
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = np.sum((gt_sig_idx == pred_sig_idx)) / gt_sig_idx.size

    if verbose:
        # Print results
        print(f"TPR: {tpr:.2f}")
        print(f"TNR: {tnr:.2f}")
        print(f"FPR: {fpr:.2f}")
        print(f"FNR: {fnr:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"F1: {f1:.2f}")
        print(f"Accuracy: {accuracy:.2f}")
        print()
    results = {
        "f1": f1, "accuracy": accuracy, "recall": recall, "precision": precision,
        "tpr": tpr, "tnr": tnr, "fpr": fpr, "fnr": fnr
    }
    return results


if __name__ == '__main__':
    np.random.seed(0)
    log_batch_factor = 1.
    nx = 1000
    ny = 1000
    n_genes = 1500
    z1 = np.random.binomial(1, 0.1, (1, n_genes))
    # mu1 = 100 + np.abs(np.random.normal(0, 5, (1, n_genes))) * z1
    mu1 = 100 + np.random.normal(15, 5, (1, n_genes)) * z1
    mu1 = np.tile(mu1, (nx, 1))
    mu2 = 100 * np.ones((ny, n_genes))

    # the first half of all samples are batch effected
    mu1[:nx // 2] *= np.exp(log_batch_factor)
    mu2[:ny // 2] *= np.exp(log_batch_factor)

    d1 = 1.
    d2 = .1

    r1 = 1 / d1
    r2 = 1 / d2
    p1 = 1 / (1 + d1 * mu1)
    p2 = 1 / (1 + d2 * mu2)

    # print("P(X = 0) = ", p1 ** r1)
    # print("P(Y = 0) = ", p2 ** r2)

    # Generate synthetic gene expression data
    X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
    Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

    # remove batch effect for visualization purposes and ground truth
    mu1[:nx // 2] *= np.exp(-log_batch_factor)
    mu2[:ny // 2] *= np.exp(-log_batch_factor)
    # mu sequences are now identical across samples within group
    true_lfcs = np.log2(mu2[0] / mu1[0]).reshape((1, n_genes))

    # normalization = { median-of-ratios | CP10K }
    sc_lfcs, sc_adj_pvals = scanpy_sig_test(X, Y, method='t-test', normalization='CP10K')
    plt.rcParams.update({'font.size': 20})
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].scatter(sc_lfcs, -np.log10(sc_adj_pvals), label="$S_1$")
    non_de_idx = true_lfcs[0] == 0
    ax[0].scatter(sc_lfcs[non_de_idx], -np.log10(sc_adj_pvals)[non_de_idx], label='True Non-DEGs', alpha=0.5, marker='x')
    ax[0].axhline(-np.log10(0.05), -1000, 1000, color='red', label='adj $p$ < 0.05')
    ax[0].set_ylabel("$-\log_{10}(p)$")
    ax[0].set_xlabel('Estimated LFC')
    ax[0].legend()
    ax[1].scatter(true_lfcs, sc_lfcs, label='$t$-test estimated LFCs')
    ax[1].scatter(true_lfcs, true_lfcs, label='True LFCs')
    ax[1].scatter(true_lfcs[0, sc_adj_pvals < 0.05], sc_lfcs[sc_adj_pvals < 0.05], color='red', label='adj $p$ < 0.05')
    ax[1].legend()
    ax[1].set_ylabel("LFC")
    ax[1].set_xlabel("LFC")
    plt.suptitle('Scanpy')
    plt.tight_layout()
    plt.show()
    print("Scanpy Test Results: ")
    sc_results = get_test_results(sc_adj_pvals, true_lfcs)


    DELN_lfcs, DELN_p_vals = get_DELN_lfcs(Y, X, test='t', normalize=True, normalization='CP10K')
    DELN_adj_pvals = smm.multipletests(DELN_p_vals, alpha=0.05, method='bonferroni')[1]
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].scatter(DELN_lfcs, -np.log10(DELN_adj_pvals), label=r'$S_\text{LN}$')
    # ax[0].scatter(true_lfcs, -np.log10(DELN_adj_pvals), label='True LFC vs DELN adj $p$')
    ax[0].scatter(DELN_lfcs[non_de_idx], -np.log10(DELN_adj_pvals)[non_de_idx], label='True Non-DEGs', alpha=0.5, marker='x')
    ax[0].axhline(-np.log10(0.05), -1000, 1000, color='red', label='adj $p$ < 0.05')
    ax[0].set_ylabel("$-\log_{10}(p)$")
    ax[0].legend()
    ax[0].set_xlabel('Estimated LFC')
    ax[1].scatter(true_lfcs, DELN_lfcs, label=r'$S_\text{LN}$')
    ax[1].scatter(true_lfcs, true_lfcs, label='True LFCs')
    ax[1].scatter(true_lfcs[0, DELN_adj_pvals < 0.05], DELN_lfcs[DELN_adj_pvals < 0.05], color='red', label='adj $p$ < 0.05')
    ax[1].set_ylabel("LFC")
    ax[1].set_xlabel("LFC")
    ax[1].legend()
    plt.suptitle("LN's $t$-test")
    plt.tight_layout()
    plt.show()
    print("LN's Test Results: ")
    deln_results = get_test_results(DELN_adj_pvals, true_lfcs)


