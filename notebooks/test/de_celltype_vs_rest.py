"""
Script to run DELN differential expression analysis for each celltype vs rest.

Usage examples:

1. Command line:
   python de_celltype_vs_rest.py --input data.h5ad --output de_results.csv

2. Programmatic:
   import scanpy as sc
   from de_celltype_vs_rest import run_de_celltype_vs_rest
   
   adata = sc.read_h5ad('data.h5ad')
   de_results = run_de_celltype_vs_rest(
       adata, 
       layer='counts', 
       celltype_col='leiden'
   )
   de_results.to_csv('de_results.csv', index=False)

3. Using test statistics (for sorting when test statistics are huge or p-values are zero):
   de_results = run_de_celltype_vs_rest(
       adata, 
       use_log_abs_statistic=True
   )
   # Sort by log_abs_test_statistic (descending) - larger values indicate stronger evidence
   # This handles cases where test statistics are huge or p-values are exactly zero
   de_results = de_results.sort_values(
       by='log_abs_test_statistic',
       ascending=False,
       na_position='last'
   )
"""

import os
import sys
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import statsmodels.stats.multitest as smm
from tqdm import tqdm

# Ensure project root (DE-ZILN) is on sys.path so local modules resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils import get_LN_lfcs as get_DELN_lfcs


def compute_fraction_expressed(counts):
    """
    Compute fraction of cells with non-zero counts for each gene.
    
    Parameters
    ----------
    counts : array-like
        Count matrix (n_cells x n_genes)
    
    Returns
    -------
    frac_expressed : ndarray
        Fraction of cells with non-zero counts for each gene
    """
    if sp.issparse(counts):
        n_nonzero = np.array((counts > 0).sum(axis=0)).flatten()
    else:
        n_nonzero = (counts > 0).sum(axis=0)
    n_cells = counts.shape[0]
    return n_nonzero / n_cells


def bonferroni_correction_log(log_p_vals, n_tests):
    """
    Apply Bonferroni correction in log space.
    
    Parameters
    ----------
    log_p_vals : array-like
        Log p-values (can be -inf for p=0)
    n_tests : int
        Number of tests
    
    Returns
    -------
    log_adj_pvals : ndarray
        Log adjusted p-values (capped at 0, i.e., log(1.0))
        -inf values remain -inf (most significant)
    """
    log_p_vals = np.asarray(log_p_vals)
    log_n_tests = np.log(n_tests)
    
    # Initialize output array
    log_adj_pvals = np.full_like(log_p_vals, -np.inf)
    
    # Handle finite values
    finite_mask = np.isfinite(log_p_vals)
    if np.any(finite_mask):
        # Bonferroni: p_adj = min(p * n_tests, 1.0)
        # In log space: log(p_adj) = min(log(p) + log(n_tests), 0)
        log_adj_pvals[finite_mask] = log_p_vals[finite_mask] + log_n_tests
        # Cap at 0 (log(1.0))
        log_adj_pvals[finite_mask] = np.minimum(log_adj_pvals[finite_mask], 0.0)
    
    # -inf values remain -inf (most significant)
    
    return log_adj_pvals


def create_pseudo_bulk(counts, batch_ids, group_size=200):
    """
    Create pseudo-bulk samples by grouping cells, ensuring cells from 
    different batches are not grouped together.
    
    Parameters
    ----------
    counts : array-like
        Count matrix (n_cells x n_genes)
    batch_ids : array-like
        Batch IDs for each cell (length n_cells)
    group_size : int
        Target number of cells per pseudo-bulk sample (default: 200)
    
    Returns
    -------
    pseudo_bulk_counts : ndarray
        Pseudo-bulk count matrix (n_pseudo_bulks x n_genes)
    n_cells_per_group : ndarray
        Number of cells in each pseudo-bulk sample
    """
    counts = np.asarray(counts) if not sp.issparse(counts) else counts.toarray()
    batch_ids = np.asarray(batch_ids)
    n_cells, n_genes = counts.shape
    
    # Get unique batches
    unique_batches = np.unique(batch_ids)
    
    pseudo_bulk_list = []
    n_cells_list = []
    
    # Process each batch separately
    for batch in unique_batches:
        batch_mask = (batch_ids == batch)
        batch_indices = np.where(batch_mask)[0]
        batch_counts = counts[batch_indices, :]
        n_batch_cells = len(batch_indices)
        
        # Calculate number of groups for this batch
        n_groups = max(1, int(np.ceil(n_batch_cells / group_size)))
        
        # Split batch cells into groups
        for i in range(n_groups):
            start_idx = i * group_size
            end_idx = min((i + 1) * group_size, n_batch_cells)
            
            if start_idx < n_batch_cells:
                # Sum counts for this group
                group_counts = batch_counts[start_idx:end_idx, :].sum(axis=0)
                pseudo_bulk_list.append(group_counts)
                n_cells_list.append(end_idx - start_idx)
    
    # Stack all pseudo-bulk samples
    pseudo_bulk_counts = np.vstack(pseudo_bulk_list)
    n_cells_per_group = np.array(n_cells_list)
    
    return pseudo_bulk_counts, n_cells_per_group


def run_de_celltype_vs_rest(adata, layer='counts', celltype_col='leiden', 
                             corr_method='bonferroni', normalize=True, 
                             normalization='CP10K', test='t',
                             use_pseudo_bulk=False, group_size=200, batch_col='batch_id',
                             use_log_abs_statistic=False):
    """
    Run DELN differential expression analysis for each celltype vs rest.
    
    Parameters
    ----------
    adata : AnnData
        Annotated data object with counts in adata.layers[layer] and 
        celltype info in adata.obs[celltype_col]
    layer : str
        Layer name containing raw counts (default: 'counts')
    celltype_col : str
        Column name in adata.obs containing celltype labels (default: 'leiden')
    corr_method : str
        Multiple testing correction method (default: 'bonferroni')
    normalize : bool
        Whether to normalize counts (default: True)
    normalization : str
        Normalization method: 'CP10K' or 'median-of-ratios' (default: 'CP10K')
    test : str
        Statistical test: 't' or 'z' (default: 't')
    use_pseudo_bulk : bool
        Whether to create pseudo-bulk samples to reduce memory (default: False)
    group_size : int
        Target number of cells per pseudo-bulk sample (default: 200)
    batch_col : str
        Column name in adata.obs containing batch/sample IDs (default: 'batch_id')
    use_log_abs_statistic : bool
        Whether to compute and return test statistics and log(|test_statistic|) 
        instead of p-values. Useful when test statistics are huge or p-values 
        are zero and cannot be sorted (default: False)
    
    Returns
    -------
    de_results : pd.DataFrame
        DataFrame with DE results for all celltypes, merged with adata.var
    """
    # Extract counts
    if layer in adata.layers:
        counts = adata.layers[layer]
    elif layer == 'X':
        counts = adata.X
    else:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available layers: {list(adata.layers.keys())}")
    
    # Get celltype labels
    if celltype_col not in adata.obs.columns:
        raise ValueError(f"Column '{celltype_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
    
    celltypes = adata.obs[celltype_col].values
    unique_celltypes = np.unique(celltypes)
    n_celltypes = len(unique_celltypes)
    n_genes = counts.shape[1]
    n_cells_original = counts.shape[0]
    
    print(f"Found {n_celltypes} celltypes: {unique_celltypes}")
    print(f"Number of genes: {n_genes}")
    print(f"Number of cells: {n_cells_original}")
    
    # Store original counts for fraction expressed calculation
    if sp.issparse(counts):
        counts_original = counts.toarray()
    else:
        counts_original = np.asarray(counts).copy()
    
    # Convert to dense array if sparse (for processing)
    if sp.issparse(counts):
        counts = counts.toarray()
    else:
        counts = np.asarray(counts)
    
    # Check if pseudo-bulking is requested
    batch_ids = None
    if use_pseudo_bulk:
        if batch_col not in adata.obs.columns:
            raise ValueError(f"Column '{batch_col}' not found in adata.obs. Available columns: {list(adata.obs.columns)}")
        
        batch_ids = adata.obs[batch_col].values
        print(f"Using pseudo-bulk samples (group_size={group_size})...")
        print(f"  Respecting batch boundaries from '{batch_col}' column")
    
    # Initialize results list
    all_results = []
    
    # Run DE for each celltype vs rest
    for celltype in tqdm(unique_celltypes, desc="Running DE analysis"):
        # Create binary mask for this celltype
        celltype_mask = (celltypes == celltype)
        rest_mask = ~celltype_mask
        
        n_celltype = np.sum(celltype_mask)
        n_rest = np.sum(rest_mask)
        
        if n_celltype == 0:
            print(f"Warning: No cells found for celltype {celltype}, skipping...")
            continue
        
        if n_rest == 0:
            print(f"Warning: No cells in rest group for celltype {celltype}, skipping...")
            continue
        
        # Extract counts for this celltype and rest (from original counts for fraction expressed)
        X_rest_original = counts_original[rest_mask, :]  # Rest group (control)
        Y_celltype_original = counts_original[celltype_mask, :]  # Celltype group (test)
        
        # Compute fraction expressed for each group (using original cell counts)
        frac_expressed_rest = compute_fraction_expressed(X_rest_original)
        frac_expressed_celltype = compute_fraction_expressed(Y_celltype_original)
        
        # Create pseudo-bulk samples if requested
        if use_pseudo_bulk:
            # Get batch IDs for this celltype and rest
            batch_ids_celltype = batch_ids[celltype_mask]
            batch_ids_rest = batch_ids[rest_mask]
            
            # Create pseudo-bulk for celltype group
            Y_celltype, _ = create_pseudo_bulk(
                Y_celltype_original, batch_ids_celltype, group_size=group_size
            )
            
            # Create pseudo-bulk for rest group
            X_rest, _ = create_pseudo_bulk(
                X_rest_original, batch_ids_rest, group_size=group_size
            )
        else:
            # Use original counts directly
            X_rest = X_rest_original
            Y_celltype = Y_celltype_original
        
        # Run DELN
        try:
            if use_log_abs_statistic:
                # Return test statistics, log(|statistic|), AND p-values
                lfcs, test_statistics, log_abs_statistics, p_vals = get_DELN_lfcs(
                    Y_celltype, X_rest, 
                    normalize=normalize, 
                    normalization=normalization, 
                    test=test,
                    return_log_abs_statistic=True
                )
                
                # Multiple testing correction
                valid_mask = ~np.isnan(p_vals)
                adj_pvals = np.full(n_genes, np.nan)
                
                if np.any(valid_mask):
                    adj_pvals[valid_mask] = smm.multipletests(
                        p_vals[valid_mask], 
                        alpha=0.05, 
                        method=corr_method
                    )[1]
            else:
                # Return p-values (original behavior)
                lfcs, p_vals = get_DELN_lfcs(
                    Y_celltype, X_rest, 
                    normalize=normalize, 
                    normalization=normalization, 
                    test=test,
                    return_log_abs_statistic=False
                )
                test_statistics = None
                log_abs_statistics = None
                
                # Multiple testing correction
                valid_mask = ~np.isnan(p_vals)
                adj_pvals = np.full(n_genes, np.nan)
                
                if np.any(valid_mask):
                    adj_pvals[valid_mask] = smm.multipletests(
                        p_vals[valid_mask], 
                        alpha=0.05, 
                        method=corr_method
                    )[1]
        except Exception as e:
            print(f"Error processing celltype {celltype}: {e}")
            # Fill with NaN if there's an error
            lfcs = np.full(n_genes, np.nan)
            if use_log_abs_statistic:
                test_statistics = np.full(n_genes, np.nan)
                log_abs_statistics = np.full(n_genes, np.nan)
                p_vals = np.full(n_genes, np.nan)
                adj_pvals = np.full(n_genes, np.nan)
            else:
                p_vals = np.full(n_genes, np.nan)
                adj_pvals = np.full(n_genes, np.nan)
                test_statistics = None
                log_abs_statistics = None
            frac_expressed_rest = np.full(n_genes, np.nan)
            frac_expressed_celltype = np.full(n_genes, np.nan)
        
        # Create results DataFrame for this celltype
        # Use gene names/index from adata.var as index
        if use_log_abs_statistic:
            # Include test statistic, log(|statistic|), AND p-values
            results_df = pd.DataFrame({
                'celltype': celltype,
                'log2_fold_change': lfcs,
                'test_statistic': test_statistics,
                'log_abs_test_statistic': log_abs_statistics,
                'abs_test_statistic': np.abs(test_statistics),  # For convenience
                'p_value': p_vals,
                'p_value_adj': adj_pvals,
                'frac_expressed_celltype': frac_expressed_celltype,
                'frac_expressed_rest': frac_expressed_rest,
                'n_cells_celltype': n_celltype,
                'n_cells_rest': n_rest
            }, index=adata.var.index)
        else:
            results_df = pd.DataFrame({
                'celltype': celltype,
                'log2_fold_change': lfcs,
                'p_value': p_vals,
                'p_value_adj': adj_pvals,
                'frac_expressed_celltype': frac_expressed_celltype,
                'frac_expressed_rest': frac_expressed_rest,
                'n_cells_celltype': n_celltype,
                'n_cells_rest': n_rest
            }, index=adata.var.index)
        
        all_results.append(results_df)
    
    # Concatenate all results
    de_results = pd.concat(all_results, ignore_index=False)
    
    # Reset index to get gene names/index as a column
    de_results = de_results.reset_index()
    gene_index_col = de_results.columns[0]  # Name of the gene index column
    
    # Merge with adata.var (reset index first to get gene names as column)
    var_df = adata.var.reset_index()
    
    # Merge on the gene index/name column
    de_results = de_results.merge(
        var_df, 
        on=gene_index_col,
        how='left',
        suffixes=('', '_var')
    )
    
    # Reorder columns to put gene info first, then DE stats
    gene_info_cols = [col for col in var_df.columns if col in de_results.columns]
    if use_log_abs_statistic:
        de_stats_cols = ['celltype', 'log2_fold_change', 'test_statistic', 'log_abs_test_statistic', 
                         'abs_test_statistic', 'p_value', 'p_value_adj',
                         'frac_expressed_celltype', 'frac_expressed_rest',
                         'n_cells_celltype', 'n_cells_rest']
    else:
        de_stats_cols = ['celltype', 'log2_fold_change', 'p_value', 'p_value_adj', 
                         'frac_expressed_celltype', 'frac_expressed_rest',
                         'n_cells_celltype', 'n_cells_rest']
    # Build ordered column list
    ordered_cols = []
    # Add gene info columns
    for col in gene_info_cols:
        if col not in ordered_cols:
            ordered_cols.append(col)
    # Add DE stats columns
    for col in de_stats_cols:
        if col in de_results.columns and col not in ordered_cols:
            ordered_cols.append(col)
    # Add any remaining columns
    for col in de_results.columns:
        if col not in ordered_cols:
            ordered_cols.append(col)
    
    de_results = de_results[ordered_cols]
    
    return de_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run DELN DE analysis for each celltype vs rest'
    )
    parser.add_argument(
        '--input', 
        type=str, 
        required=True,
        help='Path to input h5ad file'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        required=True,
        help='Path to output CSV file'
    )
    parser.add_argument(
        '--layer', 
        type=str, 
        default='counts',
        help='Layer name containing raw counts (default: counts)'
    )
    parser.add_argument(
        '--celltype_col', 
        type=str, 
        default='leiden',
        help='Column name in adata.obs containing celltype labels (default: leiden)'
    )
    parser.add_argument(
        '--corr_method', 
        type=str, 
        default='bonferroni',
        choices=['bonferroni', 'fdr_bh', 'fdr_by', 'fdr_tsbh', 'fdr_tsbky'],
        help='Multiple testing correction method (default: bonferroni)'
    )
    parser.add_argument(
        '--no_normalize', 
        action='store_true',
        help='Disable normalization (use raw counts)'
    )
    parser.add_argument(
        '--normalization', 
        type=str, 
        default='CP10K',
        choices=['CP10K', 'median-of-ratios'],
        help='Normalization method (default: CP10K)'
    )
    parser.add_argument(
        '--test', 
        type=str, 
        default='t',
        choices=['t', 'z'],
        help='Statistical test: t-test or z-test (default: t)'
    )
    parser.add_argument(
        '--use_pseudo_bulk', 
        action='store_true',
        help='Create pseudo-bulk samples to reduce memory usage (default: False)'
    )
    parser.add_argument(
        '--group_size', 
        type=int, 
        default=200,
        help='Target number of cells per pseudo-bulk sample (default: 200)'
    )
    parser.add_argument(
        '--batch_col', 
        type=str, 
        default='batch_id',
        help='Column name in adata.obs containing batch/sample IDs (default: batch_id)'
    )
    parser.add_argument(
        '--use_log_abs_statistic', 
        action='store_true',
        help='Compute and return test statistics and log(|test_statistic|) in addition to p-values. '
             'Useful when test statistics are huge or p-values are zero and cannot be sorted (default: False)'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from {args.input}...")
    adata = sc.read_h5ad(args.input)
    print(f"Data loaded. Shape: {adata.shape}")
    
    # Run DE analysis
    print("Running DE analysis...")
    if args.use_log_abs_statistic:
        print("Using test statistics and log(|test_statistic|) in addition to p-values (handles huge test statistics)")
    de_results = run_de_celltype_vs_rest(
        adata,
        layer=args.layer,
        celltype_col=args.celltype_col,
        corr_method=args.corr_method,
        normalize=not args.no_normalize,
        normalization=args.normalization,
        test=args.test,
        use_pseudo_bulk=args.use_pseudo_bulk,
        group_size=args.group_size,
        batch_col=args.batch_col,
        use_log_abs_statistic=args.use_log_abs_statistic
    )
    
    # Save results
    print(f"Saving results to {args.output}...")
    de_results.to_csv(args.output, index=False)
    print(f"Done! Results saved to {args.output}")
    print(f"Total rows: {len(de_results)}")
    print(f"Number of celltypes: {de_results['celltype'].nunique()}")

