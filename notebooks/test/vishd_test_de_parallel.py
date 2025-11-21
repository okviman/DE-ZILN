import os
import sys
import numpy as np
import statsmodels.stats.multitest as smm
import scanpy as sc
import scipy.sparse as sp
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd

# Ensure project root (DE-ZILN) is on sys.path so local modules resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_LN_lfcs as get_DELN_lfcs
from scanpy_ttest import scanpy_sig_test


def de_test_single(X, Y, selected_genes, true_signs):
    """Run DE test on X and Y, return TP, FP, FN, TN counts and rates for all methods."""
    n_genes = X.shape[-1]

    idx_X = set(np.arange(n_genes)[X.sum(0) == 0])
    idx_Y = set(np.arange(n_genes)[Y.sum(0) == 0])
    union_unexpressed_gene_set = idx_X.union(idx_Y)
    cols_to_remove = np.array(list(union_unexpressed_gene_set))
    mask = np.ones(n_genes, dtype=bool)
    mask[cols_to_remove] = False
    X = X[:, mask].copy()
    Y = Y[:, mask].copy()

    # Update selected_genes indices after removing unexpressed genes
    # Map original gene indices to new indices
    original_to_new = {}
    new_idx = 0
    for orig_idx in range(n_genes):
        if mask[orig_idx]:
            original_to_new[orig_idx] = new_idx
            new_idx += 1
    
    selected_genes_filtered = set([original_to_new[g] for g in selected_genes if g in original_to_new])
    true_signs_filtered = {original_to_new[g]: true_signs[g] for g in selected_genes if g in original_to_new}

    n_genes_filtered = X.shape[-1]
    n_selected = len(selected_genes_filtered)
    n_not_selected = n_genes_filtered - n_selected
    results = {}

    for method in ["DELN", "t-test", 'wilcoxon']:
        if method == "DELN":
            method_key = "LN_test"  # Rename DELN to LN_test in output
        else:
            method_key = "Scanpy " + method
        if method == "DELN":
            lfcs, DELN_p_vals = get_DELN_lfcs(Y, X, test='t')
            adj_pvals = smm.multipletests(DELN_p_vals, alpha=0.05, method='bonferroni')[1]
        else:
            lfcs, adj_pvals = scanpy_sig_test(X, Y, method=method)
            # Convert pandas Series to numpy arrays if needed
            if hasattr(adj_pvals, 'values'):
                adj_pvals = adj_pvals.values
            if hasattr(lfcs, 'values'):
                lfcs = lfcs.values
        
        # Convert to numpy array and handle NaN values
        adj_pvals = np.asarray(adj_pvals).flatten()
        lfcs = np.asarray(lfcs).flatten()
        
        # Ensure arrays have the correct length
        if len(adj_pvals) != n_genes_filtered or len(lfcs) != n_genes_filtered:
            raise ValueError(f"Array length mismatch: adj_pvals={len(adj_pvals)}, lfcs={len(lfcs)}, expected={n_genes_filtered}")
        
        # Find significant genes (adj_pval < 0.05, treating NaN as not significant)
        # NaN < 0.05 evaluates to False, which is correct
        significant_mask = (adj_pvals < 0.05) & (~np.isnan(adj_pvals))
        
        # Create boolean arrays for selected genes
        selected_mask = np.zeros(n_genes_filtered, dtype=bool)
        selected_mask[list(selected_genes_filtered)] = True
        
        # True Positives: significant AND in selected_genes
        tp = np.sum(significant_mask & selected_mask)
        
        # False Positives: significant AND NOT in selected_genes
        fp = np.sum(significant_mask & (~selected_mask))
        
        # False Negatives: NOT significant AND in selected_genes
        fn = n_selected - tp
        
        # True Negatives: NOT significant AND NOT in selected_genes
        tn = n_not_selected - fp
        
        # Compute rates as ratios
        # TPR (True Positive Rate / Sensitivity) = TP / (TP + FN) = TP / n_selected
        tpr = tp / n_selected if n_selected > 0 else 0.0
        tpr = np.clip(tpr, 0.0, 1.0)  # Ensure in [0, 1]
        
        # FPR (False Positive Rate) = FP / (FP + TN) = FP / n_not_selected
        fpr = fp / n_not_selected if n_not_selected > 0 else 0.0
        fpr = np.clip(fpr, 0.0, 1.0)  # Ensure in [0, 1]
        
        # FNR (False Negative Rate) = FN / (TP + FN) = FN / n_selected
        fnr = fn / n_selected if n_selected > 0 else 0.0
        fnr = np.clip(fnr, 0.0, 1.0)  # Ensure in [0, 1]
        
        # TNR (True Negative Rate / Specificity) = TN / (FP + TN) = TN / n_not_selected
        tnr = tn / n_not_selected if n_not_selected > 0 else 0.0
        tnr = np.clip(tnr, 0.0, 1.0)  # Ensure in [0, 1]
        
        results[method_key] = {
            "tp": tp, 
            "fp": fp, 
            "fn": fn,
            "tn": tn,
            "n_selected": n_selected,
            "n_total": n_genes_filtered,
            "tpr": tpr,
            "fpr": fpr,
            "fnr": fnr,
            "tnr": tnr
        }
    
    return results


def run_single_iteration(args):
    """Run a single iteration of the test with given parameters."""
    p, q, lfc, n_cells_remove, umis_base, library_sizes_base, seed = args
    
    # Set random seed for reproducibility within each iteration
    np.random.seed(seed)
    
    # Create a copy of the base data
    umis = umis_base.copy()
    library_sizes = library_sizes_base.copy()
    
    # Split cells by median depth (before downsampling)
    median_libsize = np.median(library_sizes)
    high_group_mask = library_sizes >= median_libsize
    low_group_mask = library_sizes < median_libsize
    
    # Randomly select genes with probability q
    n_genes = umis.shape[1]
    selected_genes = set(np.where(np.random.random(n_genes) < q)[0])
    
    # Randomly assign signs (+1 or -1) to selected genes
    true_signs = {g: np.random.choice([-1, 1]) for g in selected_genes}
    
    # Apply fold changes
    umis_modified = umis.copy()
    for gene_idx in selected_genes:
        sign = true_signs[gene_idx]
        fold_change = 2**lfc
        
        if sign == 1:
            # Multiply counts in HIGH group
            umis_modified[high_group_mask, gene_idx] = np.round(
                umis_modified[high_group_mask, gene_idx] * fold_change
            ).astype(int)
        else:  # sign == -1
            # Multiply counts in LOW group
            umis_modified[low_group_mask, gene_idx] = np.round(
                umis_modified[low_group_mask, gene_idx] * fold_change
            ).astype(int)
    
    # Downsample counts for each cell for each gene by ratio p
    # Using binomial downsampling (skip if p == 1.0)
    if p < 1.0:
        umis_modified = np.random.binomial(umis_modified.astype(int), p).astype(float)
    
    # Recalculate library sizes after downsampling
    library_sizes_modified = umis_modified.sum(1)
    
    # Split again after downsampling
    median_libsize_modified = np.median(library_sizes_modified)
    X = umis_modified[library_sizes_modified >= median_libsize_modified]
    Y = umis_modified[library_sizes_modified < median_libsize_modified]
    
    # Run the test
    return de_test_single(X, Y, selected_genes, true_signs)


def run_tests_for_p(p, q, lfc, n_cells_remove, umis_base, library_sizes_base, n_reps, n_jobs):
    """Run n_reps iterations for a given p value in parallel."""
    # Create arguments for each iteration
    seeds = np.random.randint(0, 2**31, size=n_reps)
    args_list = [(p, q, lfc, n_cells_remove, umis_base, library_sizes_base, seed) for seed in seeds]
    
    # Run in parallel
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(run_single_iteration, args_list)
    
    # Aggregate results
    aggregated = defaultdict(lambda: {"tpr": [], "fpr": [], "fnr": [], "tnr": []})
    for result in results_list:
        for method, metrics in result.items():
            aggregated[method]["tpr"].append(metrics["tpr"])
            aggregated[method]["fpr"].append(metrics["fpr"])
            aggregated[method]["fnr"].append(metrics["fnr"])
            aggregated[method]["tnr"].append(metrics["tnr"])
    
    # Compute means and stds of the ratios
    summary = {}
    for method, metrics in aggregated.items():
        tpr_values = np.array(metrics["tpr"])
        fpr_values = np.array(metrics["fpr"])
        fnr_values = np.array(metrics["fnr"])
        tnr_values = np.array(metrics["tnr"])
        
        summary[method] = {
            "tpr_mean": np.mean(tpr_values),
            "tpr_std": np.std(tpr_values),
            "fpr_mean": np.mean(fpr_values),
            "fpr_std": np.std(fpr_values),
            "fnr_mean": np.mean(fnr_values),
            "fnr_std": np.std(fnr_values),
            "tnr_mean": np.mean(tnr_values),
            "tnr_std": np.std(tnr_values),
        }
    
    return summary


def save_results(p_values, results_by_p, output_file, q, lfc):
    """Save DE test results to a CSV file."""
    # Prepare data for DataFrame
    rows = []
    for p in p_values:
        for method, metrics in results_by_p[p].items():
            rows.append({
                'p': p,
                'q': q,
                'lfc': lfc,
                'method': method,
                'tpr_mean': metrics['tpr_mean'],
                'tpr_std': metrics['tpr_std'],
                'fpr_mean': metrics['fpr_mean'],
                'fpr_std': metrics['fpr_std'],
                'fnr_mean': metrics['fnr_mean'],
                'fnr_std': metrics['fnr_std'],
                'tnr_mean': metrics['tnr_mean'],
                'tnr_std': metrics['tnr_std']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def plot_results(p_values, results_by_p, output_file, q, lfc):
    """Plot TPR, FPR, FNR, and TNR vs p for all methods."""
    methods = list(results_by_p[p_values[0]].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot TPR
    for idx, method in enumerate(methods):
        tpr_means = [results_by_p[p][method]["tpr_mean"] for p in p_values]
        tpr_stds = [results_by_p[p][method]["tpr_std"] for p in p_values]
        
        axes[0, 0].errorbar(p_values, tpr_means, yerr=tpr_stds, 
                        marker='o', capsize=5, capthick=2, label=method, 
                        color=colors[idx], linewidth=2, markersize=6)
    
    axes[0, 0].set_xlabel(r'Downsampling ratio $p$', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axes[0, 0].set_title('TPR vs Downsample Ratio', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(loc='best', fontsize=10)
    axes[0, 0].set_ylim([0, 1])
    
    # Plot FPR
    for idx, method in enumerate(methods):
        fpr_means = [results_by_p[p][method]["fpr_mean"] for p in p_values]
        fpr_stds = [results_by_p[p][method]["fpr_std"] for p in p_values]
        
        axes[0, 1].errorbar(p_values, fpr_means, yerr=fpr_stds, 
                        marker='o', capsize=5, capthick=2, label=method, 
                        color=colors[idx], linewidth=2, markersize=6)
    
    axes[0, 1].set_xlabel(r'Downsampling ratio $p$', fontsize=12)
    axes[0, 1].set_ylabel('False Positive Rate (FPR)', fontsize=12)
    axes[0, 1].set_title('FPR vs Downsample Ratio', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(loc='best', fontsize=10)
    axes[0, 1].set_ylim([0, 1])
    
    # Plot FNR
    for idx, method in enumerate(methods):
        fnr_means = [results_by_p[p][method]["fnr_mean"] for p in p_values]
        fnr_stds = [results_by_p[p][method]["fnr_std"] for p in p_values]
        
        axes[1, 0].errorbar(p_values, fnr_means, yerr=fnr_stds, 
                        marker='o', capsize=5, capthick=2, label=method, 
                        color=colors[idx], linewidth=2, markersize=6)
    
    axes[1, 0].set_xlabel(r'Downsampling ratio $p$', fontsize=12)
    axes[1, 0].set_ylabel('False Negative Rate (FNR)', fontsize=12)
    axes[1, 0].set_title('FNR vs Downsample Ratio', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(loc='best', fontsize=10)
    axes[1, 0].set_ylim([0, 1])
    
    # Plot TNR
    for idx, method in enumerate(methods):
        tnr_means = [results_by_p[p][method]["tnr_mean"] for p in p_values]
        tnr_stds = [results_by_p[p][method]["tnr_std"] for p in p_values]
        
        axes[1, 1].errorbar(p_values, tnr_means, yerr=tnr_stds, 
                        marker='o', capsize=5, capthick=2, label=method, 
                        color=colors[idx], linewidth=2, markersize=6)
    
    axes[1, 1].set_xlabel(r'Downsampling ratio $p$', fontsize=12)
    axes[1, 1].set_ylabel('True Negative Rate (TNR)', fontsize=12)
    axes[1, 1].set_title(f'TNR vs Downsample Ratio (q={q}, lfc={lfc})', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='best', fontsize=10)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DE detection with parallel execution')
    parser.add_argument('--n_cells_remove', type=int, required=True,
                        help='Number of cells with lowest total count to remove')
    parser.add_argument('--q', type=float, required=True,
                        help='Fraction of genes to introduce DE')
    parser.add_argument('--lfc', type=float, required=True,
                        help='Log2 fold change to apply')
    parser.add_argument('--p_min', type=float, default=0.1,
                        help='Minimum downsample ratio p (default: 0.1)')
    parser.add_argument('--p_max', type=float, default=1.0,
                        help='Maximum downsample ratio p (default: 1.0)')
    parser.add_argument('--p_steps', type=int, default=10,
                        help='Number of p values to test (default: 10)')
    parser.add_argument('--n_reps', type=int, default=100,
                        help='Number of repetitions per p value (default: 100)')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs (default: all available cores)')
    parser.add_argument('--output', type=str, default='de_test_plot.png',
                        help='Output file for the plot (default: de_test_plot.png)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Output file for results CSV (default: auto-generated from output name)')
    args = parser.parse_args()
    
    n_cells_remove = args.n_cells_remove
    q = args.q
    lfc = args.lfc
    p_min = args.p_min
    p_max = args.p_max
    p_steps = args.p_steps
    n_reps = args.n_reps
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    output_file = args.output
    
    # Generate results file name if not provided
    if args.results_file is None:
        base_name = os.path.splitext(output_file)[0]
        results_file = f"{base_name}_results.csv"
    else:
        results_file = args.results_file
    
    if not (0 < p_min <= p_max <= 1.0):
        raise ValueError("p_min and p_max must be between 0 and 1 (inclusive), and p_min <= p_max")
    
    if not (0 < q <= 1.0):
        raise ValueError("q must be between 0 and 1")
    
    print(f"Loading data...")
    # Load spatial ovary data (raw read counts in adata.X)
    h5ad = sc.read_h5ad("../../spatial-ovary/notebooks/oskar/merged_blobs_in_cluster_5.h5ad")
    # Make gene names unique using gene_id
    h5ad.var = h5ad.var.reset_index().set_index('gene_ids')
    # Robustly convert to a dense numpy array
    X_raw = h5ad.X
    if sp.issparse(X_raw):
        umis_w_zeros = X_raw.toarray()
    else:
        umis_w_zeros = np.asarray(X_raw)

    print(f"Preprocessing data...")
    # preprocessing
    # remove genes with zero counts accross all cells
    idx = umis_w_zeros.sum(0) > 0
    umis = umis_w_zeros[:, idx]
    n_genes = umis.shape[-1]
    n_cells = umis.shape[0]

    # remove the n_cells_remove cells with lowest read counts
    library_sizes = umis.sum(1)
    order = np.argsort(library_sizes)
    if order.size > n_cells_remove:
        keep_idx = order[n_cells_remove:]
        umis_base = umis[keep_idx].copy()
        library_sizes_base = library_sizes[keep_idx].copy()
    else:
        umis_base = umis.copy()
        library_sizes_base = library_sizes.copy()
    
    print(f"Data loaded. Shape: {umis_base.shape}")
    print(f"Parameters: q={q}, lfc={lfc}")
    print(f"Running tests for {p_steps} p values, {n_reps} repetitions each, using {n_jobs} cores...")
    
    # Generate p values
    p_values = np.linspace(p_min, p_max, p_steps)
    
    # Run tests for each p value
    results_by_p = {}
    for i, p in enumerate(p_values):
        print(f"Processing p={p:.3f} ({i+1}/{p_steps})...")
        results_by_p[p] = run_tests_for_p(p, q, lfc, n_cells_remove, umis_base, library_sizes_base, n_reps, n_jobs)
        # Print summary for this p
        for method, metrics in results_by_p[p].items():
            print(f"  {method}: TPR={metrics['tpr_mean']:.4f}±{metrics['tpr_std']:.4f}, "
                  f"FPR={metrics['fpr_mean']:.4f}±{metrics['fpr_std']:.4f}, "
                  f"FNR={metrics['fnr_mean']:.4f}±{metrics['fnr_std']:.4f}, "
                  f"TNR={metrics['tnr_mean']:.4f}±{metrics['tnr_std']:.4f}")
    
    print(f"\nSaving results...")
    save_results(p_values, results_by_p, results_file, q, lfc)
    
    print(f"Generating plots...")
    # plot_results(p_values, results_by_p, output_file, q, lfc)
    print("Done!")

