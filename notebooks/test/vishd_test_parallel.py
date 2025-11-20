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
from scanpy_ttest import get_test_results, scanpy_sig_test


def fpr_test_single(X, Y):
    """Run FPR test on X and Y, return results for all methods."""
    n_genes = X.shape[-1]

    idx_X = set(np.arange(n_genes)[X.sum(0) == 0])
    idx_Y = set(np.arange(n_genes)[Y.sum(0) == 0])
    union_unexpressed_gene_set = idx_X.union(idx_Y)
    cols_to_remove = np.array(list(union_unexpressed_gene_set))
    mask = np.ones(n_genes, dtype=bool)
    mask[cols_to_remove] = False
    X = X[:, mask].copy()
    Y = Y[:, mask].copy()

    n_genes = X.shape[-1]
    results = {}

    for method in ["DELN", "t-test", 'wilcoxon']:
        if method == "DELN":
            method_key = "LN_test"  # Rename DELN to LN_test in output
        else:
            method_key = "Scanpy " + method
        if method == "DELN":
            lfcs, DELN_p_vals = get_DELN_lfcs(Y, X, test='t')
            adj_pvals = smm.multipletests(DELN_p_vals, alpha=0.05, method='bonferroni')[1]
            if np.sum(adj_pvals >= 0.05) == n_genes:
                results[method_key] = {"fpr": 0., "fpr_filtered": 0.}
            else:
                test_results = get_test_results(adj_pvals, np.zeros(n_genes), verbose=False)
                fpr_filtered = np.sum(np.abs(lfcs[adj_pvals < 0.05]) > 0.25) / adj_pvals.size
                results[method_key] = {"fpr": test_results['fpr'], "fpr_filtered": fpr_filtered}
        else:
            lfcs, adj_pvals = scanpy_sig_test(X, Y, method=method)
            if np.sum(adj_pvals >= 0.05) == n_genes:
                results[method_key] = {"fpr": 0., "fpr_filtered": 0.}
            else:
                test_results = get_test_results(adj_pvals, np.zeros(n_genes), verbose=False)
                fpr_filtered = np.sum(np.abs(lfcs[adj_pvals < 0.05]) > 0.25) / adj_pvals.size
                results[method_key] = {"fpr": test_results['fpr'], "fpr_filtered": fpr_filtered}
    
    return results


def run_single_iteration(args):
    """Run a single iteration of the test with given parameters."""
    p, n_cells_remove, umis_base, library_sizes_base, seed = args
    
    # Set random seed for reproducibility within each iteration
    np.random.seed(seed)
    
    # Create a copy of the base data
    umis = umis_base.copy()
    library_sizes = library_sizes_base.copy()
    
    # Downsample counts for each cell for each gene by ratio p
    # Using binomial downsampling (skip if p == 1.0)
    if p < 1.0:
        umis = np.random.binomial(umis.astype(int), p).astype(float)
    # else: p == 1.0, no downsampling needed
    
    # Recalculate library sizes after downsampling
    library_sizes = umis.sum(1)

    # split remaining cells by median depth
    median_libsize = np.median(library_sizes)
    X = umis[library_sizes >= median_libsize]
    Y = umis[library_sizes < median_libsize]
    
    # Run the test
    return fpr_test_single(X, Y)


def run_tests_for_p(p, n_cells_remove, umis_base, library_sizes_base, n_reps, n_jobs):
    """Run n_reps iterations for a given p value in parallel."""
    # Create arguments for each iteration
    seeds = np.random.randint(0, 2**31, size=n_reps)
    args_list = [(p, n_cells_remove, umis_base, library_sizes_base, seed) for seed in seeds]
    
    # Run in parallel
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(run_single_iteration, args_list)
    
    # Aggregate results
    aggregated = defaultdict(lambda: {"fpr": [], "fpr_filtered": []})
    for result in results_list:
        for method, metrics in result.items():
            aggregated[method]["fpr"].append(metrics["fpr"])
            aggregated[method]["fpr_filtered"].append(metrics["fpr_filtered"])
    
    # Compute means and stds (both raw and log10)
    summary = {}
    for method, metrics in aggregated.items():
        fpr_values = np.array(metrics["fpr"])
        fpr_filtered_values = np.array(metrics["fpr_filtered"])
        
        # Raw statistics
        fpr_mean = np.mean(fpr_values)
        fpr_std = np.std(fpr_values)
        fpr_filtered_mean = np.mean(fpr_filtered_values)
        fpr_filtered_std = np.std(fpr_filtered_values)
        
        # Log10 statistics (handle zeros by adding small epsilon)
        epsilon = 1e-10
        fpr_log10_values = np.log10(fpr_values + epsilon)
        fpr_filtered_log10_values = np.log10(fpr_filtered_values + epsilon)
        
        fpr_log10_mean = np.mean(fpr_log10_values)
        fpr_log10_std = np.std(fpr_log10_values)
        fpr_filtered_log10_mean = np.mean(fpr_filtered_log10_values)
        fpr_filtered_log10_std = np.std(fpr_filtered_log10_values)
        
        summary[method] = {
            "fpr_mean": fpr_mean,
            "fpr_std": fpr_std,
            "fpr_filtered_mean": fpr_filtered_mean,
            "fpr_filtered_std": fpr_filtered_std,
            "fpr_log10_mean": fpr_log10_mean,
            "fpr_log10_std": fpr_log10_std,
            "fpr_filtered_log10_mean": fpr_filtered_log10_mean,
            "fpr_filtered_log10_std": fpr_filtered_log10_std,
            "fpr_values": fpr_values.tolist(),  # Save raw values for later analysis
            "fpr_filtered_values": fpr_filtered_values.tolist()
        }
    
    return summary


def save_results(p_values, results_by_p, output_file):
    """Save FPR results to a CSV file."""
    # Prepare data for DataFrame
    rows = []
    for p in p_values:
        for method, metrics in results_by_p[p].items():
            rows.append({
                'p': p,
                'method': method,
                'fpr_mean': metrics['fpr_mean'],
                'fpr_std': metrics['fpr_std'],
                'fpr_filtered_mean': metrics['fpr_filtered_mean'],
                'fpr_filtered_std': metrics['fpr_filtered_std'],
                'fpr_log10_mean': metrics['fpr_log10_mean'],
                'fpr_log10_std': metrics['fpr_log10_std'],
                'fpr_filtered_log10_mean': metrics['fpr_filtered_log10_mean'],
                'fpr_filtered_log10_std': metrics['fpr_filtered_log10_std']
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def plot_fdr_results(p_values, results_by_p, output_file):
    """Plot log10(FPR) and log10(FPR Filtered) vs p for all methods in the same plot."""
    methods = list(results_by_p[p_values[0]].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot FPR
    for idx, method in enumerate(methods):
        fpr_log10_means = [results_by_p[p][method]["fpr_log10_mean"] for p in p_values]
        fpr_log10_stds = [results_by_p[p][method]["fpr_log10_std"] for p in p_values]
        
        axes[0].errorbar(p_values, fpr_log10_means, yerr=fpr_log10_stds, 
                        marker='o', capsize=5, capthick=2, label=method, 
                        color=colors[idx], linewidth=2, markersize=6)
    
    axes[0].set_xlabel('Downsample ratio (p)', fontsize=12)
    axes[0].set_ylabel('log10(FPR)', fontsize=12)
    axes[0].set_title('FPR vs Downsample Ratio', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    
    # Plot FPR Filtered
    for idx, method in enumerate(methods):
        fpr_filtered_log10_means = [results_by_p[p][method]["fpr_filtered_log10_mean"] for p in p_values]
        fpr_filtered_log10_stds = [results_by_p[p][method]["fpr_filtered_log10_std"] for p in p_values]
        
        axes[1].errorbar(p_values, fpr_filtered_log10_means, yerr=fpr_filtered_log10_stds, 
                        marker='o', capsize=5, capthick=2, label=method, 
                        color=colors[idx], linewidth=2, markersize=6)
    
    axes[1].set_xlabel('Downsample ratio (p)', fontsize=12)
    axes[1].set_ylabel('log10(FPR Filtered)', fontsize=12)
    axes[1].set_title('FPR Filtered (|lfc| > 0.25) vs Downsample Ratio', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test FPR for different DE methods with parallel execution')
    parser.add_argument('--n_cells_remove', type=int, required=True,
                        help='Number of cells with lowest total count to remove')
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
    parser.add_argument('--output', type=str, default='fdr_plot.png',
                        help='Output file for the plot (default: fdr_plot.png)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Output file for results CSV (default: auto-generated from output name)')
    args = parser.parse_args()
    
    n_cells_remove = args.n_cells_remove
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
    print(f"Running tests for {p_steps} p values, {n_reps} repetitions each, using {n_jobs} cores...")
    
    # Generate p values
    p_values = np.linspace(p_min, p_max, p_steps)
    
    # Run tests for each p value
    results_by_p = {}
    for i, p in enumerate(p_values):
        print(f"Processing p={p:.3f} ({i+1}/{p_steps})...")
        results_by_p[p] = run_tests_for_p(p, n_cells_remove, umis_base, library_sizes_base, n_reps, n_jobs)
        # Print summary for this p
        for method, metrics in results_by_p[p].items():
            print(f"  {method}: FPR={metrics['fpr_mean']:.4f}±{metrics['fpr_std']:.4f}, "
                  f"FPR Filtered={metrics['fpr_filtered_mean']:.4f}±{metrics['fpr_filtered_std']:.4f}")
    
    print(f"\nSaving results...")
    save_results(p_values, results_by_p, results_file)
    
    print(f"Generating plots...")
    # plot_fdr_results(p_values, results_by_p, output_file)
    print("Done!")

