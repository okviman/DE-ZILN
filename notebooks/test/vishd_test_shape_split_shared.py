import os
import sys
import numpy as np
import statsmodels.stats.multitest as smm
import scanpy as sc
import scipy.sparse as sp
import argparse
from multiprocessing import Pool, cpu_count, shared_memory
from functools import partial
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import json
from sklearn.metrics import average_precision_score, precision_recall_curve, auc

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
        significant_mask = (adj_pvals < 0.05) & (~np.isnan(adj_pvals))
        
        # Create boolean arrays for selected genes
        selected_mask = np.zeros(n_genes_filtered, dtype=bool)
        selected_mask[list(selected_genes_filtered)] = True
        
        # Prepare labels and scores for PR metrics
        y_true = selected_mask.astype(int)
        # Use adjusted p-values to form scores: smaller p -> larger score
        epsilon = 1e-300
        pvals_for_score = adj_pvals.copy()
        # Replace non-finite with worst (1.0)
        invalid_mask = ~np.isfinite(pvals_for_score)
        if np.any(invalid_mask):
            pvals_for_score[invalid_mask] = 1.0
        pvals_for_score = np.clip(pvals_for_score, epsilon, 1.0)
        y_scores = -np.log10(pvals_for_score)
        # Compute AP and trapezoidal PR-AUC if both classes present
        if y_true.sum() > 0 and y_true.sum() < y_true.size:
            try:
                ap = average_precision_score(y_true, y_scores)
                precision_pr, recall, _ = precision_recall_curve(y_true, y_scores)
                pr_auc = auc(recall, precision_pr)
            except Exception:
                ap = np.nan
                pr_auc = np.nan
                precision_pr = np.array([1.0])
                recall = np.array([0.0, 1.0])
        else:
            ap = np.nan
            pr_auc = np.nan
            precision_pr = np.array([1.0])
            recall = np.array([0.0, 1.0])
        
        # Compute precision-recall curves across thresholds (based on gene ranking)
        # Sort genes by score (descending: best genes first)
        sorted_indices = np.argsort(y_scores)[::-1]  # Descending order
        sorted_y_true = y_true[sorted_indices]
        
        # Compute cumulative TP, FP, FN at each threshold
        # At threshold k, we consider top k genes as "significant"
        n_total = len(y_true)
        n_positive = y_true.sum()
        
        # Initialize arrays for precision and recall curves
        precision_curve = []
        recall_curve = []
        
        # Start with threshold 0 (no genes selected)
        tp_cum = 0
        fp_cum = 0
        
        # Add point at threshold 0
        # When no predictions: precision = 1.0 (by convention), recall = 0.0
        precision_curve.append(1.0)
        recall_curve.append(0.0)
        
        # Compute at each threshold (top k genes)
        for k in range(1, n_total + 1):
            # Check if this gene is a true positive
            if sorted_y_true[k-1] == 1:
                tp_cum += 1
            else:
                fp_cum += 1
            
            # Compute metrics at this threshold
            fn_cum = n_positive - tp_cum
            
            # Precision = TP / (TP + FP)
            prec = tp_cum / (tp_cum + fp_cum) if (tp_cum + fp_cum) > 0 else 1.0
            
            # Recall = TP / (TP + FN) = TP / n_positive
            rec = tp_cum / n_positive if n_positive > 0 else 0.0
            
            precision_curve.append(prec)
            recall_curve.append(rec)
        
        precision_curve = np.array(precision_curve)
        recall_curve = np.array(recall_curve)
        
        # Compute area under precision-recall curve (PR-AUC)
        # For PR curves, we need to ensure monotonicity: as recall increases,
        # we should use the maximum precision achieved at that recall level
        # Sort by recall, then use maximum precision at each recall level
        if len(recall_curve) > 1:
            # Sort by recall (ascending)
            recall_sorted_idx = np.argsort(recall_curve)
            recall_sorted = recall_curve[recall_sorted_idx]
            precision_sorted = precision_curve[recall_sorted_idx]
            
            # For PR curves, we want to use the maximum precision at each recall level
            # This creates a monotonic decreasing precision curve as recall increases
            # Reverse the arrays and compute cumulative maximum precision
            # (going from high recall to low recall, precision should be non-decreasing)
            recall_reversed = recall_sorted[::-1]
            precision_reversed = precision_sorted[::-1]
            
            # Compute cumulative maximum precision (backwards from high to low recall)
            precision_max = np.maximum.accumulate(precision_reversed)
            
            # Reverse back to get precision as a function of increasing recall
            recall_final = recall_reversed[::-1]
            precision_final = precision_max[::-1]
            
            # Compute AUC using trapezoidal rule
            pr_auc_curve = np.trapz(precision_final, recall_final)
        else:
            pr_auc_curve = np.nan
        
        # Also compute metrics at fixed threshold (adj_pval < 0.05) for backward compatibility
        # True Positives: significant AND in selected_genes
        tp = np.sum(significant_mask & selected_mask)
        
        # False Positives: significant AND NOT in selected_genes
        fp = np.sum(significant_mask & (~selected_mask))
        
        # False Negatives: NOT significant AND in selected_genes
        fn = n_selected - tp
        
        # True Negatives: NOT significant AND NOT in selected_genes
        tn = n_not_selected - fp
        
        # Compute rates as ratios
        tpr = tp / n_selected if n_selected > 0 else 0.0
        tpr = np.clip(tpr, 0.0, 1.0)
        
        fpr = fp / n_not_selected if n_not_selected > 0 else 0.0
        fpr = np.clip(fpr, 0.0, 1.0)
        
        fnr = fn / n_selected if n_selected > 0 else 0.0
        fnr = np.clip(fnr, 0.0, 1.0)
        
        tnr = tn / n_not_selected if n_not_selected > 0 else 0.0
        tnr = np.clip(tnr, 0.0, 1.0)
        
        # Compute accuracy: (TP + TN) / (TP + TN + FP + FN)
        accuracy = (tp + tn) / n_genes_filtered if n_genes_filtered > 0 else 0.0
        accuracy = np.clip(accuracy, 0.0, 1.0)
        
        # Compute precision: TP / (TP + FP) - only defined when there are significant genes
        n_significant = tp + fp
        precision = tp / n_significant if n_significant > 0 else 0.0
        precision = np.clip(precision, 0.0, 1.0)
        
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
            "tnr": tnr,
            "accuracy": accuracy,
            "precision": precision,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "pr_auc_curve": pr_auc_curve,  # Area under precision-recall curve (from ranking)
            "ap": ap,
            "pr_auc": pr_auc  # PR-AUC from sklearn (for comparison)
        }
    
    return results


def create_groups_from_shape_ids_shared(umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, p, seed=None):
    """
    Split spots within each shape_id into two groups A and B using shared memory.
    
    For each shape_id:
    1. Sample n_A from Binomial(n_spots, p)
    2. Randomly select n_A spots for group A, rest for group B
    3. Sum reads for each group to create two cells
    
    Parameters:
    -----------
    umis_shm_name : str
        Name of shared memory block for umis array
    umis_shape : tuple
        Shape of umis array (n_spots, n_genes)
    umis_dtype : numpy.dtype
        Data type of umis array
    shape_id_to_indices : dict
        Pre-computed mapping from shape_id to array of spot indices
    unique_shape_ids : array
        Pre-computed unique shape IDs (in order)
    p : float
        Probability for group A (0 < p <= 0.5)
    seed : int, optional
        Random seed
        
    Returns:
    --------
    X : array (n_shape_ids, n_genes)
        Summed counts for group A (one cell per shape_id)
    Y : array (n_shape_ids, n_genes)
        Summed counts for group B (one cell per shape_id)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Attach to shared memory
    existing_shm = shared_memory.SharedMemory(name=umis_shm_name)
    umis = np.ndarray(umis_shape, dtype=umis_dtype, buffer=existing_shm.buf)
    
    n_shape_ids = len(unique_shape_ids)
    n_genes = umis.shape[1]
    
    X = np.zeros((n_shape_ids, n_genes), dtype=umis_dtype)
    Y = np.zeros((n_shape_ids, n_genes), dtype=umis_dtype)
    
    for idx, shape_id in enumerate(unique_shape_ids):
        # Get pre-computed spot indices for this shape_id
        spot_indices = shape_id_to_indices[shape_id]
        n_spots = len(spot_indices)
        
        # Sample n_A from Binomial(n_spots, p)
        # Ensure at least 1 spot in A and at least 1 in B
        n_A = np.random.binomial(n_spots, p)
        n_A = max(1, min(n_A, n_spots - 1))  # At least 1 in A, at least 1 in B
        
        # Optimized random selection: shuffle and slice (faster than choice for large arrays)
        # Create a copy to avoid modifying the original
        shuffled_indices = spot_indices.copy()
        np.random.shuffle(shuffled_indices)
        
        # Split into A and B groups
        group_A_indices = shuffled_indices[:n_A]
        group_B_indices = shuffled_indices[n_A:]
        
        # Sum reads for each group
        X[idx, :] = umis[group_A_indices, :].sum(axis=0)
        Y[idx, :] = umis[group_B_indices, :].sum(axis=0)
    
    # Close shared memory connection (but don't unlink - main process handles that)
    existing_shm.close()
    
    return X, Y


def run_single_iteration_fpr(args):
    """Run a single iteration of the FPR test with given parameters."""
    p, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, seed = args
    
    # Create groups from shape_ids (uses shared memory, does splitting in worker)
    X, Y = create_groups_from_shape_ids_shared(umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, p, seed=seed)
    
    # Run the test
    return fpr_test_single(X, Y)


def run_single_iteration_de(args):
    """Run a single iteration of the DE test with given parameters."""
    p, q, lfc, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, seed = args
    
    # Set random seed for reproducibility within each iteration
    np.random.seed(seed)
    
    # Create groups from shape_ids (uses shared memory, does splitting in worker)
    X, Y = create_groups_from_shape_ids_shared(umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, p, seed=seed)
    
    # Randomly select genes with probability q
    n_genes = X.shape[1]
    selected_genes = set(np.where(np.random.random(n_genes) < q)[0])
    
    # Randomly assign signs (+1 or -1) to selected genes
    true_signs = {g: np.random.choice([-1, 1]) for g in selected_genes}
    
    # Apply fold changes
    X_modified = X.copy()
    Y_modified = Y.copy()
    
    effective_selected_genes = set()
    effective_true_signs = {}
    for gene_idx in selected_genes:
        sign = true_signs[gene_idx]
        fold_change = 2**lfc
        x_sum = X[:, gene_idx].sum()
        y_sum = Y[:, gene_idx].sum()
        
        # If both groups are zero for this gene, skip it
        if x_sum == 0 and y_sum == 0:
            continue
        
        # If intended up in Y but Y is all zero and X has counts, flip to up in X
        if sign == 1 and y_sum == 0 and x_sum > 0:
            sign = -1
        # If intended up in X (sign == -1) but X is all zero and Y has counts, flip to up in Y
        elif sign == -1 and x_sum == 0 and y_sum > 0:
            sign = 1
        
        if sign == 1:
            # Multiply counts in group Y
            Y_modified[:, gene_idx] = np.round(Y_modified[:, gene_idx] * fold_change).astype(int)
        else:  # sign == -1
            # Multiply counts in group X
            X_modified[:, gene_idx] = np.round(X_modified[:, gene_idx] * fold_change).astype(int)
        
        effective_selected_genes.add(gene_idx)
        effective_true_signs[gene_idx] = sign
    
    # Run the test
    return de_test_single(X_modified, Y_modified, effective_selected_genes, effective_true_signs)


def run_tests_for_p_fpr(p, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, n_reps, n_jobs):
    """Run n_reps iterations for a given p value in parallel (FPR mode)."""
    # Create arguments for each iteration - workers do the splitting
    seeds = np.random.randint(0, 2**31, size=n_reps)
    args_list = [(p, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, seed) for seed in seeds]
    
    # Run in parallel
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(run_single_iteration_fpr, args_list)
    
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
        
        # Log10 statistics: compute log10 for EACH iteration, then compute mean and std
        # This is important: mean(log10(FPR)) != log10(mean(FPR))
        # We compute log10 per iteration to get correct statistics for plotting
        epsilon = 1e-10
        fpr_log10_values = np.log10(fpr_values + epsilon)  # log10 for each iteration
        fpr_filtered_log10_values = np.log10(fpr_filtered_values + epsilon)  # log10 for each iteration
        
        fpr_log10_mean = np.mean(fpr_log10_values)  # mean of log10 values
        fpr_log10_std = np.std(fpr_log10_values)  # std of log10 values
        fpr_filtered_log10_mean = np.mean(fpr_filtered_log10_values)  # mean of log10 values
        fpr_filtered_log10_std = np.std(fpr_filtered_log10_values)  # std of log10 values
        
        summary[method] = {
            "fpr_mean": fpr_mean,
            "fpr_std": fpr_std,
            "fpr_filtered_mean": fpr_filtered_mean,
            "fpr_filtered_std": fpr_filtered_std,
            "fpr_log10_mean": fpr_log10_mean,
            "fpr_log10_std": fpr_log10_std,
            "fpr_filtered_log10_mean": fpr_filtered_log10_mean,
            "fpr_filtered_log10_std": fpr_filtered_log10_std,
            "fpr_values": fpr_values.tolist(),
            "fpr_filtered_values": fpr_filtered_values.tolist()
        }
    
    return summary


def run_tests_for_p_de(p, q, lfc, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, n_reps, n_jobs):
    """Run n_reps iterations for a given p value in parallel (DE test mode)."""
    # Create arguments for each iteration - workers do the splitting
    seeds = np.random.randint(0, 2**31, size=n_reps)
    args_list = [(p, q, lfc, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, seed) for seed in seeds]
    
    # Run in parallel
    with Pool(processes=n_jobs) as pool:
        results_list = pool.map(run_single_iteration_de, args_list)
    
    # Aggregate results
    aggregated = defaultdict(lambda: {"tpr": [], "fpr": [], "fnr": [], "tnr": [], "accuracy": [], "precision": [], 
                                      "precision_curve": [], "recall_curve": [], "pr_auc_curve": [], "ap": [], "pr_auc": []})
    for result in results_list:
        for method, metrics in result.items():
            aggregated[method]["tpr"].append(metrics["tpr"])
            aggregated[method]["fpr"].append(metrics["fpr"])
            aggregated[method]["fnr"].append(metrics["fnr"])
            aggregated[method]["tnr"].append(metrics["tnr"])
            aggregated[method]["accuracy"].append(metrics["accuracy"])
            aggregated[method]["precision"].append(metrics["precision"])
            aggregated[method]["precision_curve"].append(metrics["precision_curve"])
            aggregated[method]["recall_curve"].append(metrics["recall_curve"])
            aggregated[method]["pr_auc_curve"].append(metrics["pr_auc_curve"])
            aggregated[method]["ap"].append(metrics["ap"])
            aggregated[method]["pr_auc"].append(metrics["pr_auc"])
    
    # Compute means and stds of the ratios
    summary = {}
    for method, metrics in aggregated.items():
        tpr_values = np.array(metrics["tpr"])
        fpr_values = np.array(metrics["fpr"])
        fnr_values = np.array(metrics["fnr"])
        tnr_values = np.array(metrics["tnr"])
        accuracy_values = np.array(metrics["accuracy"])
        precision_values = np.array(metrics["precision"])
        
        ap_values = np.array(metrics["ap"], dtype=float)
        pr_auc_values = np.array(metrics["pr_auc"], dtype=float)
        pr_auc_curve_values = np.array(metrics["pr_auc_curve"], dtype=float)
        
        # Aggregate precision-recall curves by interpolating to common recall values
        # Use a common set of recall values from 0 to 1
        recall_grid = np.linspace(0, 1, 101)  # 101 points from 0 to 1
        precision_interpolated = []
        
        for prec_curve, rec_curve in zip(metrics["precision_curve"], metrics["recall_curve"]):
            # Sort by recall for interpolation
            sort_idx = np.argsort(rec_curve)
            rec_sorted = rec_curve[sort_idx]
            prec_sorted = prec_curve[sort_idx]
            
            # Interpolate precision at common recall values
            # Use forward fill for values outside the range
            prec_interp = np.interp(recall_grid, rec_sorted, prec_sorted, 
                                  left=prec_sorted[0] if len(prec_sorted) > 0 else 1.0,
                                  right=prec_sorted[-1] if len(prec_sorted) > 0 else 0.0)
            precision_interpolated.append(prec_interp)
        
        precision_interpolated = np.array(precision_interpolated)
        precision_curve_mean = np.mean(precision_interpolated, axis=0)
        precision_curve_std = np.std(precision_interpolated, axis=0)
        
        summary[method] = {
            "tpr_mean": np.mean(tpr_values),
            "tpr_std": np.std(tpr_values),
            "fpr_mean": np.mean(fpr_values),
            "fpr_std": np.std(fpr_values),
            "fnr_mean": np.mean(fnr_values),
            "fnr_std": np.std(fnr_values),
            "tnr_mean": np.mean(tnr_values),
            "tnr_std": np.std(tnr_values),
            "accuracy_mean": np.mean(accuracy_values),
            "accuracy_std": np.std(accuracy_values),
            "precision_mean": np.mean(precision_values),
            "precision_std": np.std(precision_values),
            "precision_curve_mean": precision_curve_mean,
            "precision_curve_std": precision_curve_std,
            "recall_grid": recall_grid,
            "pr_auc_curve_mean": np.nanmean(pr_auc_curve_values),
            "pr_auc_curve_std": np.nanstd(pr_auc_curve_values),
            "ap_mean": np.nanmean(ap_values),
            "ap_std": np.nanstd(ap_values),
            "pr_auc_mean": np.nanmean(pr_auc_values),
            "pr_auc_std": np.nanstd(pr_auc_values),
        }
    
    return summary


def save_results_fpr(p_values, results_by_p, output_file):
    """Save FPR results to a CSV file."""
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


def save_results_de(p_values, results_by_p, output_file, q, lfc):
    """Save DE test results to a CSV file."""
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
                'tnr_std': metrics['tnr_std'],
                'accuracy_mean': metrics.get('accuracy_mean', np.nan),
                'accuracy_std': metrics.get('accuracy_std', np.nan),
                'precision_mean': metrics.get('precision_mean', np.nan),
                'precision_std': metrics.get('precision_std', np.nan),
                'pr_auc_curve_mean': metrics.get('pr_auc_curve_mean', np.nan),
                'pr_auc_curve_std': metrics.get('pr_auc_curve_std', np.nan),
                'ap_mean': metrics.get('ap_mean', np.nan),
                'ap_std': metrics.get('ap_std', np.nan),
                'pr_auc_mean': metrics.get('pr_auc_mean', np.nan),
                'pr_auc_std': metrics.get('pr_auc_std', np.nan)
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")


def save_pr_curve_data(p_values, results_by_p, output_file, q, lfc):
    """Save PR curve data (precision and recall arrays) for later plotting."""
    curve_data = {
        'q': q,
        'lfc': lfc,
        'p_values': p_values.tolist(),
        'curves': {}
    }
    
    for p in p_values:
        p_key = f"p_{p:.6f}"  # Use high precision to avoid rounding issues
        curve_data['curves'][p_key] = {}
        
        for method, metrics in results_by_p[p].items():
            recall_grid = metrics.get("recall_grid", None)
            precision_curve_mean = metrics.get("precision_curve_mean", None)
            precision_curve_std = metrics.get("precision_curve_std", None)
            pr_auc_curve_mean = metrics.get("pr_auc_curve_mean", np.nan)
            pr_auc_curve_std = metrics.get("pr_auc_curve_std", np.nan)
            
            if recall_grid is not None and precision_curve_mean is not None:
                curve_data['curves'][p_key][method] = {
                    'recall': recall_grid.tolist(),
                    'precision_mean': precision_curve_mean.tolist(),
                    'precision_std': precision_curve_std.tolist() if precision_curve_std is not None else None,
                    'pr_auc_curve_mean': float(pr_auc_curve_mean) if not np.isnan(pr_auc_curve_mean) else None,
                    'pr_auc_curve_std': float(pr_auc_curve_std) if not np.isnan(pr_auc_curve_std) else None
                }
    
    with open(output_file, 'w') as f:
        json.dump(curve_data, f, indent=2)
    print(f"PR curve data saved to {output_file}")


def plot_fpr_results(p_values, results_by_p, output_file):
    """Plot log10(FPR) and log10(FPR Filtered) vs p for all methods."""
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
    
    axes[0].set_xlabel('Split probability (p)', fontsize=12)
    axes[0].set_ylabel('log10(FPR)', fontsize=12)
    axes[0].set_title('FPR vs Split Probability', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    
    # Plot FPR Filtered
    for idx, method in enumerate(methods):
        fpr_filtered_log10_means = [results_by_p[p][method]["fpr_filtered_log10_mean"] for p in p_values]
        fpr_filtered_log10_stds = [results_by_p[p][method]["fpr_filtered_log10_std"] for p in p_values]
        
        axes[1].errorbar(p_values, fpr_filtered_log10_means, yerr=fpr_filtered_log10_stds, 
                        marker='o', capsize=5, capthick=2, label=method, 
                        color=colors[idx], linewidth=2, markersize=6)
    
    axes[1].set_xlabel('Split probability (p)', fontsize=12)
    axes[1].set_ylabel('log10(FPR Filtered)', fontsize=12)
    axes[1].set_title('FPR Filtered (|lfc| > 0.25) vs Split Probability', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_de_results(p_values, results_by_p, output_file, q, lfc):
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
    
    axes[0, 0].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[0, 0].set_ylabel('True Positive Rate (TPR)', fontsize=12)
    axes[0, 0].set_title('TPR vs Split Probability', fontsize=14, fontweight='bold')
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
    
    axes[0, 1].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[0, 1].set_ylabel('False Positive Rate (FPR)', fontsize=12)
    axes[0, 1].set_title('FPR vs Split Probability', fontsize=14, fontweight='bold')
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
    
    axes[1, 0].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[1, 0].set_ylabel('False Negative Rate (FNR)', fontsize=12)
    axes[1, 0].set_title('FNR vs Split Probability', fontsize=14, fontweight='bold')
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
    
    axes[1, 1].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[1, 1].set_ylabel('True Negative Rate (TNR)', fontsize=12)
    axes[1, 1].set_title(f'TNR vs Split Probability (q={q}, lfc={lfc})', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(loc='best', fontsize=10)
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_pr_results(p_values, results_by_p, output_file):
    """Plot Average Precision (AP) and PR-AUC vs p for all methods."""
    methods = list(results_by_p[p_values[0]].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot AP
    for idx, method in enumerate(methods):
        ap_means = [results_by_p[p][method].get("ap_mean", np.nan) for p in p_values]
        ap_stds = [results_by_p[p][method].get("ap_std", np.nan) for p in p_values]
        axes[0].errorbar(p_values, ap_means, yerr=ap_stds,
                         marker='o', capsize=5, capthick=2, label=method,
                         color=colors[idx], linewidth=2, markersize=6)
    axes[0].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[0].set_ylabel('Average Precision (AP)', fontsize=12)
    axes[0].set_title('AP vs Split Probability', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc='best', fontsize=10)
    axes[0].set_ylim([0, 1])
    
    # Plot PR-AUC (trapezoidal)
    for idx, method in enumerate(methods):
        pr_auc_means = [results_by_p[p][method].get("pr_auc_mean", np.nan) for p in p_values]
        pr_auc_stds = [results_by_p[p][method].get("pr_auc_std", np.nan) for p in p_values]
        axes[1].errorbar(p_values, pr_auc_means, yerr=pr_auc_stds,
                         marker='o', capsize=5, capthick=2, label=method,
                         color=colors[idx], linewidth=2, markersize=6)
    axes[1].set_xlabel(r'Split probability $p$', fontsize=12)
    axes[1].set_ylabel('PR-AUC (Trapezoidal)', fontsize=12)
    axes[1].set_title('PR-AUC vs Split Probability', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc='best', fontsize=10)
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


def plot_precision_recall_results(p_values, results_by_p, output_file, q, lfc):
    """Plot Precision vs Recall curves for each p value (PR curves based on gene ranking)."""
    methods = list(results_by_p[p_values[0]].keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
    
    # Create subplots: one for each p value
    n_p = len(p_values)
    n_cols = min(3, n_p)
    n_rows = (n_p + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
    if n_p == 1:
        axes = [axes]
    else:
        axes = axes.flatten() if n_rows > 1 else axes
    
    for p_idx, p in enumerate(p_values):
        ax = axes[p_idx]
        
        for method_idx, method in enumerate(methods):
            recall_grid = results_by_p[p][method].get("recall_grid", None)
            precision_curve_mean = results_by_p[p][method].get("precision_curve_mean", None)
            precision_curve_std = results_by_p[p][method].get("precision_curve_std", None)
            pr_auc_curve_mean = results_by_p[p][method].get("pr_auc_curve_mean", np.nan)
            
            if recall_grid is not None and precision_curve_mean is not None:
                # Plot mean curve
                ax.plot(recall_grid, precision_curve_mean, 
                       label=f'{method} (AUC={pr_auc_curve_mean:.3f})',
                       color=colors[method_idx], linewidth=2)
                
                # Plot std as shaded region
                if precision_curve_std is not None:
                    ax.fill_between(recall_grid, 
                                  precision_curve_mean - precision_curve_std,
                                  precision_curve_mean + precision_curve_std,
                                  alpha=0.2, color=colors[method_idx])
        
        ax.set_xlabel('Recall', fontsize=11)
        ax.set_ylabel('Precision', fontsize=11)
        ax.set_title(f'Precision vs Recall (p={p:.3f}, q={q}, lfc={lfc})', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
    
    # Hide unused subplots
    for idx in range(n_p, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test DE methods using shape_id-based spot splitting with shared memory')
    parser.add_argument('--n_shape_ids_remove', type=int, required=True,
                        help='Number of shape_ids with lowest total count to remove')
    parser.add_argument('--q', type=float, default=0.0,
                        help='Fraction of genes to introduce DE (0 for FPR test, >0 for DE test, default: 0.0)')
    parser.add_argument('--lfc', type=float, default=1.0,
                        help='Log2 fold change to apply (only used if q > 0, default: 1.0)')
    parser.add_argument('--p_min', type=float, default=0.1,
                        help='Minimum split probability p (default: 0.1)')
    parser.add_argument('--p_max', type=float, default=0.5,
                        help='Maximum split probability p (default: 0.5)')
    parser.add_argument('--p_steps', type=int, default=10,
                        help='Number of p values to test (default: 10)')
    parser.add_argument('--n_reps', type=int, default=100,
                        help='Number of repetitions per p value (default: 100)')
    parser.add_argument('--n_jobs', type=int, default=None,
                        help='Number of parallel jobs (default: all available cores)')
    parser.add_argument('--output', type=str, default='shape_split_test_plot.png',
                        help='Output file for the plot (default: shape_split_test_plot.png)')
    parser.add_argument('--results_file', type=str, default=None,
                        help='Output file for results CSV (default: auto-generated from output name)')
    parser.add_argument('--input_file', type=str, 
                        default='../../spatial-ovary/notebooks/oskar/merged_blobs_in_cluster_5.h5ad',
                        help='Input h5ad file path (default: ../../spatial-ovary/notebooks/oskar/merged_blobs_in_cluster_5.h5ad)')
    args = parser.parse_args()
    
    n_shape_ids_remove = args.n_shape_ids_remove
    q = args.q
    lfc = args.lfc
    p_min = args.p_min
    p_max = args.p_max
    p_steps = args.p_steps
    n_reps = args.n_reps
    n_jobs = args.n_jobs if args.n_jobs is not None else cpu_count()
    output_file = args.output
    input_file = args.input_file
    
    # Generate results file name if not provided
    if args.results_file is None:
        base_name = os.path.splitext(output_file)[0]
        results_file = f"{base_name}_results.csv"
    else:
        results_file = args.results_file
    
    if not (0 < p_min <= p_max <= 0.5):
        raise ValueError("p_min and p_max must be between 0 and 0.5 (inclusive), and p_min <= p_max")
    
    if not (0 <= q <= 1.0):
        raise ValueError("q must be between 0 and 1")
    
    if q == 0:
        print("Running in FPR test mode (q=0, no DE genes introduced)")
    else:
        print(f"Running in DE test mode (q={q}, lfc={lfc})")
    
    print(f"Loading data from {input_file}...")
    # Load spatial ovary data (raw read counts in adata.X)
    h5ad = sc.read_h5ad(input_file)
    
    # Check if shape_id column exists
    if 'shape_id' not in h5ad.obs.columns:
        raise ValueError("adata.obs must contain a 'shape_id' column")
    
    # Make gene names unique using gene_id
    if 'gene_ids' in h5ad.var.columns:
        h5ad.var = h5ad.var.reset_index().set_index('gene_ids')
    
    # Robustly convert to a dense numpy array
    X_raw = h5ad.X
    if sp.issparse(X_raw):
        umis_w_zeros = X_raw.toarray()
    else:
        umis_w_zeros = np.asarray(X_raw)
    
    shape_ids = h5ad.obs['shape_id'].values
    
    print(f"Preprocessing data...")
    # preprocessing
    # remove genes with zero counts across all spots
    idx = umis_w_zeros.sum(0) > 0
    umis = umis_w_zeros#[:, idx]
    n_genes = umis.shape[-1]
    n_spots = umis.shape[0]
    
    print(f"Original data: {n_spots} spots, {n_genes} genes")
    print(f"Number of unique shape_ids: {len(np.unique(shape_ids))}")
    
    # Remove shape_ids with lowest total counts
    # Compute total counts per shape_id
    unique_shape_ids = np.unique(shape_ids)
    shape_id_counts = {}
    for shape_id in unique_shape_ids:
        shape_mask = (shape_ids == shape_id)
        shape_id_counts[shape_id] = umis[shape_mask, :].sum()
    
    # Sort shape_ids by total count
    sorted_shape_ids = sorted(shape_id_counts.items(), key=lambda x: x[1])
    
    # Remove the n_shape_ids_remove shape_ids with lowest counts
    if len(sorted_shape_ids) > n_shape_ids_remove:
        removed_shape_ids = set([sid for sid, _ in sorted_shape_ids[:n_shape_ids_remove]])
        keep_mask = np.array([sid not in removed_shape_ids for sid in shape_ids])
        umis_base = umis[keep_mask, :].copy()
        shape_ids_base = shape_ids[keep_mask].copy()
    else:
        umis_base = umis.copy()
        shape_ids_base = shape_ids.copy()
    
    n_spots_after = umis_base.shape[0]
    n_shape_ids_after = len(np.unique(shape_ids_base))
    print(f"After removing {n_shape_ids_remove} shape_ids: {n_spots_after} spots, {n_shape_ids_after} shape_ids")
    print(f"Data shape: {umis_base.shape}")
    
    # Convert sparse to dense once for efficiency (if sparse)
    # This avoids repeated sparse operations
    if sp.issparse(umis_base):
        print("Converting sparse matrix to dense for faster processing...")
        umis_base = umis_base.toarray()
        print("Conversion complete.")
    
    # Pre-compute shape_id to indices mapping (once, used for all iterations)
    print("Pre-computing shape_id to indices mapping...")
    unique_shape_ids = np.unique(shape_ids_base)
    shape_id_to_indices = {}
    for shape_id in unique_shape_ids:
        shape_id_to_indices[shape_id] = np.where(shape_ids_base == shape_id)[0]
    print(f"Pre-computed mapping for {len(unique_shape_ids)} shape_ids")
    
    # Create shared memory for umis_base to avoid copying to each worker
    # This is critical for large arrays (e.g., 600K spots × 18K genes)
    print("Creating shared memory for umis_base to avoid copying to workers...")
    umis_shape = umis_base.shape
    umis_dtype = umis_base.dtype
    shm = shared_memory.SharedMemory(create=True, size=umis_base.nbytes)
    umis_shm = np.ndarray(umis_shape, dtype=umis_dtype, buffer=shm.buf)
    umis_shm[:] = umis_base[:]  # Copy data into shared memory
    umis_shm_name = shm.name
    print(f"Shared memory created: {shm.name}, size: {umis_base.nbytes / 1e9:.2f} GB")
    
    print(f"Running tests for {p_steps} p values, {n_reps} repetitions each, using {n_jobs} cores...")
    print("Note: Splitting is done in parallel in worker processes, testing is also parallelized.")
    
    # Generate p values
    p_values = np.linspace(p_min, p_max, p_steps)
    
    # Run tests for each p value
    results_by_p = {}
    try:
        for i, p in enumerate(p_values):
            print(f"Processing p={p:.3f} ({i+1}/{p_steps})...")
            if q == 0:
                results_by_p[p] = run_tests_for_p_fpr(p, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, n_reps, n_jobs)
                # Print summary for this p
                for method, metrics in results_by_p[p].items():
                    print(f"  {method}: FPR={metrics['fpr_mean']:.4f}±{metrics['fpr_std']:.4f}, "
                          f"FPR Filtered={metrics['fpr_filtered_mean']:.4f}±{metrics['fpr_filtered_std']:.4f}")
            else:
                results_by_p[p] = run_tests_for_p_de(p, q, lfc, umis_shm_name, umis_shape, umis_dtype, shape_id_to_indices, unique_shape_ids, n_reps, n_jobs)
                # Print summary for this p
                for method, metrics in results_by_p[p].items():
                    print(f"  {method}: "
                          f"AP={metrics.get('ap_mean', np.nan):.4f}±{metrics.get('ap_std', np.nan):.4f}, "
                          f"PR-AUC={metrics.get('pr_auc_mean', np.nan):.4f}±{metrics.get('pr_auc_std', np.nan):.4f}, "
                          f"PR-AUC-Curve={metrics.get('pr_auc_curve_mean', np.nan):.4f}±{metrics.get('pr_auc_curve_std', np.nan):.4f}, "
                          f"Accuracy={metrics.get('accuracy_mean', np.nan):.4f}±{metrics.get('accuracy_std', np.nan):.4f}, "
                          f"Precision={metrics.get('precision_mean', np.nan):.4f}±{metrics.get('precision_std', np.nan):.4f}, "
                          f"TPR={metrics['tpr_mean']:.4f}±{metrics['tpr_std']:.4f}, "
                          f"FPR={metrics['fpr_mean']:.4f}±{metrics['fpr_std']:.4f}, "
                          f"FNR={metrics['fnr_mean']:.4f}±{metrics['fnr_std']:.4f}, "
                          f"TNR={metrics['tnr_mean']:.4f}±{metrics['tnr_std']:.4f}")
    
    finally:
        # Clean up shared memory
        print("Cleaning up shared memory...")
        shm.close()
        shm.unlink()  # Release the shared memory block
    
    print(f"\nSaving results...")
    if q == 0:
        save_results_fpr(p_values, results_by_p, results_file)
        # plot_fpr_results(p_values, results_by_p, output_file)
    else:
        save_results_de(p_values, results_by_p, results_file, q, lfc)
        # Save PR curve data for later plotting
        base_name = os.path.splitext(output_file)[0]
        pr_curve_data_file = f"{base_name}_pr_curve_data.json"
        save_pr_curve_data(p_values, results_by_p, pr_curve_data_file, q, lfc)
        # plot_de_results(p_values, results_by_p, output_file, q, lfc)
        # Also plot PR metrics (AP and PR-AUC)
        # pr_output_file = f"{base_name}_pr.png"
        # plot_pr_results(p_values, results_by_p, pr_output_file)
        # Also plot precision-recall curves (based on gene ranking)
        # pr_curve_output_file = f"{base_name}_pr_curve.png"
        # plot_precision_recall_results(p_values, results_by_p, pr_curve_output_file, q, lfc)
    print("Done!")

