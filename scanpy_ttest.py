import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.stats.multitest as smm
from sklearn.metrics import confusion_matrix
from utils import get_DELN_lfcs
import matplotlib.pyplot as plt


def scanpy_sig_test(X, Y, method='t-test'):
    nx = X.shape[0]
    ny = Y.shape[0]
    n_genes = Y.shape[1]
    # Create Scanpy AnnData Object
    X_group = np.repeat("X", nx)  # Labels for group X
    Y_group = np.repeat("Y", ny)  # Labels for group Y

    adata = sc.AnnData(np.vstack([X, Y]))  # Combine X and Y
    adata.var_names = [f"Gene{i}" for i in range(n_genes)]  # Gene names
    adata.obs["group"] = np.concatenate([X_group, Y_group])  # Assign group labels

    adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    # Run Differential Expression Analysis using Scanpy's t-test
    sc.tl.rank_genes_groups(adata, groupby="group", method=method, reference="X")

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


def generate_latex_table(results1, results2, caption="Comparison of Results", label="tab:results"):
    """
    Generates a LaTeX table comparing two dictionaries of results.

    Parameters:
    - results1: dict of performance metrics
    - results2: dict of performance metrics
    - caption: LaTeX table caption
    - label: LaTeX label for referencing
    """
    metrics = results1.keys()  # Assuming both dicts have the same keys

    latex_table = [
        "\\begin{table}[h]",
        "    \\centering",
        "    \\begin{tabular}{|l|c|c|}",
        "        \\hline",
        "        Metric & Model 1 & Model 2 \\\\",
        "        \\hline"
    ]

    for metric in metrics:
        latex_table.append(f"        {metric} & {results1[metric]:.2f} & {results2[metric]:.2f} \\\\")

    latex_table.extend([
        "        \\hline",
        "    \\end{tabular}",
        f"    \\caption{{{caption}}}",
        f"    \\label{{{label}}}",
        "\\end{table}"
    ])

    return "\n".join(latex_table)


if __name__ == '__main__':
    np.random.seed(0)
    log_batch_factor = 1.
    nx = 1000
    ny = 1000
    n_genes = 1500
    mu1 = 10 + np.abs(np.random.normal(0, 5, (1, n_genes))) * np.random.binomial(1, 0.1, (1, n_genes))
    mu2 = 10 * np.exp(log_batch_factor)
    d1 = 1.
    d2 = 1.

    r1 = 1 / d1
    r2 = 1 / d2
    p1 = 1 / (1 + d1 * mu1)
    p2 = 1 / (1 + d2 * mu2)

    # print("P(X = 0) = ", p1 ** r1)
    # print("P(Y = 0) = ", p2 ** r2)

    # Generate synthetic gene expression data
    X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
    Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

    true_lfcs = np.log2((mu2 * np.exp(-log_batch_factor)) / mu1)

    sc_lfcs, sc_adj_pvals = scanpy_sig_test(X, Y, method='t-test')
    plt.rcParams.update({'font.size': 20})
    # sc_adj_pvals = smm.multipletests(sc_p_vals, alpha=0.05, method='fdr_bh')[1]
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].scatter(sc_lfcs, -np.log10(sc_adj_pvals), label="Scanpy Estimates")
    non_de_idx = true_lfcs[0] == 0
    ax[0].scatter(sc_lfcs[non_de_idx], -np.log10(sc_adj_pvals)[non_de_idx], label='True Non-DEGs', alpha=0.5, marker='x')
    # ax[0].scatter(true_lfcs, -np.log10(sc_adj_pvals))
    ax[0].axhline(-np.log10(0.05), -1000, 1000, color='red', label='adj $p$ < 0.05')
    ax[0].set_ylabel("$-\log_{10}(p)$")
    ax[0].set_xlabel('Estimated LFC')
    ax[0].legend()
    ax[1].scatter(true_lfcs, sc_lfcs, label='Estimated LFCs')
    ax[1].scatter(true_lfcs, true_lfcs, label='True LFCs')
    ax[1].scatter(true_lfcs[0, sc_adj_pvals < 0.05], sc_lfcs[sc_adj_pvals < 0.05], color='red', label='DE Classified')
    ax[1].legend()
    ax[1].set_ylabel("LFC")
    ax[1].set_xlabel("LFC")
    plt.suptitle('Scanpy')
    plt.tight_layout()
    plt.show()
    print("Scanpy Test Results: ")
    sc_results = get_test_results(sc_adj_pvals, true_lfcs)


    DELN_lfcs, DELN_p_vals = get_DELN_lfcs(Y, X, test='t')
    DELN_adj_pvals = smm.multipletests(DELN_p_vals, alpha=0.05, method='fdr_bh')[1]
    _, ax = plt.subplots(1, 2, figsize=(20, 10))
    ax[0].scatter(DELN_lfcs, -np.log10(DELN_adj_pvals), label='DELN Estimates')
    # ax[0].scatter(true_lfcs, -np.log10(DELN_adj_pvals), label='True LFC vs DELN adj $p$')
    ax[0].scatter(DELN_lfcs[non_de_idx], -np.log10(DELN_adj_pvals)[non_de_idx], label='True Non-DEGs', alpha=0.5, marker='x')
    ax[0].axhline(-np.log10(0.05), -1000, 1000, color='red', label='adj $p$ < 0.05')
    ax[0].set_ylabel("$-\log_{10}(p)$")
    ax[0].legend()
    ax[0].set_xlabel('Estimated LFC')
    ax[1].scatter(true_lfcs, DELN_lfcs, label='Estimated LFCs')
    ax[1].scatter(true_lfcs, true_lfcs, label='True LFCs')
    ax[1].scatter(true_lfcs[0, DELN_adj_pvals < 0.05], DELN_lfcs[DELN_adj_pvals < 0.05], color='red', label='DE Classified')
    ax[1].set_ylabel("LFC")
    ax[1].set_xlabel("LFC")
    ax[1].legend()
    plt.suptitle('DELN')
    plt.tight_layout()
    plt.show()
    print("DELN Test Results: ")
    deln_results = get_test_results(DELN_adj_pvals, true_lfcs)

    print(generate_latex_table(deln_results, sc_results))

