#!/usr/bin/env python3
"""
Script to perform differential expression analysis between two specified cell types.

This script runs DE using both LN test and Scanpy's t-test (and wilcoxon) and MAST,
saving all results to the updated AnnData file.

Usage:
    python run_celltype_de.py --config config.yaml [--output-dir output/]
"""

import argparse
import os
import sys
import yaml
import numpy as np
import scanpy as sc
import scipy.sparse as sp
import pandas as pd
from rpy2.robjects import pandas2ri, numpy2ri, r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

def load_scanpy_wrapper():
    """Load scanpy_wrapper module."""
    import importlib.util
    file_path = os.path.join(
        os.path.dirname(__file__), '../../pkg/scanpy_wrapper.py'
    )
    spec = importlib.util.spec_from_file_location("scanpy_wrapper", file_path)
    scanpy_wrapper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(scanpy_wrapper)
    return scanpy_wrapper

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    required_fields = ['adata_path', 'celltype_column', 'celltype_values']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    if not isinstance(config['celltype_values'], (list, tuple)) or len(config['celltype_values']) != 2:
        raise ValueError("celltype_values must be a list of two values: [celltype_1, celltype_2]")
    return config

def tocsr_if_not_sparse(x):
    """Convert ndarray/dense matrix to CSR sparse format if not already sparse."""
    if sp.issparse(x):
        return x.copy()
    return sp.csr_matrix(x)

def run_mast_de(
    adata,
    group_col,
    group_1,
    group_2,
    log1p_layer="log1p_norm",
    key_added="mast"
):
    """
    Run MAST differential expression analysis via rpy2 on a Scanpy AnnData object,
    comparing only two groups.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing single cell data.
    group_col : str
        Name of the obs column giving group labels.
    group_1, group_2 : str
        The two group values to compare.
    log1p_layer : str
        Layer in adata to use for log1p-normalized expression (default: "log1p_norm").
    key_added : str, optional
        Key to store the results in adata.uns. If None, defaults to "mast".
    """
    mast = importr('MAST')
    base = importr('base')
    stats = importr('stats')

    # Restrict to only the cells for the two groups
    mask = adata.obs[group_col].isin([group_1, group_2])
    ad = adata[mask].copy()
    X = ad.layers[log1p_layer]
    if sp.issparse(X):
        X = X.toarray()
    cell_names = list(ad.obs_names)
    gene_names = list(ad.var_names)
    n_cells, n_genes = X.shape
    assert n_cells == len(cell_names)
    assert n_genes == len(gene_names)

    # Prepare group vector ("target" vs "rest") for MAST
    group_labels = ad.obs[group_col].values
    # Assign group_1 -> "target", group_2 -> "rest" (or vice versa)
    binary_labels = np.where(group_labels == group_1, "target", "rest")

    # Matrix to R
    X_T = np.asfortranarray(X.T)  # genes x cells
    with localconverter(ro.default_converter + numpy2ri.converter):
        r_expr = ro.r.matrix(
            ro.FloatVector(X_T.ravel(order='F')),
            nrow=n_genes, ncol=n_cells, byrow=False
        )
    r_expr.rownames = ro.StrVector(gene_names)
    r_expr.colnames = ro.StrVector(cell_names)
    ro.globalenv['expr_mat'] = r_expr

    cell_data = pd.DataFrame({
        'wellKey': cell_names,
        'group': binary_labels
    }, index=cell_names)
    with localconverter(ro.default_converter + pandas2ri.converter):
        r_cell_data = ro.conversion.py2rpy(cell_data)
    ro.globalenv['cell_data'] = r_cell_data

    ro.r('''
    library(MAST)
    library(data.table)
    sca <- FromMatrix(
        exprsArray = expr_mat,
        cData = data.frame(cell_data),
        fData = data.frame(primerid = rownames(expr_mat))
    )
    colData(sca)$group <- factor(colData(sca)$group, levels = c("rest", "target"))
    cdr <- colSums(assay(sca) > 0)
    colData(sca)$cngeneson <- scale(cdr)
    zlmCond <- zlm(~ group + cngeneson, sca)
    summaryCond <- summary(zlmCond, doLRT = "grouptarget")
    summaryDt <- summaryCond$datatable
    fcHurdle <- merge(
        summaryDt[component == "H", .(primerid, `Pr(>Chisq)`)],
        summaryDt[component == "logFC" & contrast == "grouptarget", .(primerid, coef)],
        by = "primerid"
    )
    colnames(fcHurdle) <- c("names", "pvals", "logfoldchanges")
    fcHurdle$pvals_adj <- p.adjust(fcHurdle$pvals, method = "BH")
    fcHurdle$scores <- -log10(fcHurdle$pvals + 1e-300) * sign(fcHurdle$logfoldchanges)
    mast_result <- fcHurdle
    rm(sca, zlmCond, summaryCond, summaryDt, fcHurdle)
    gc()
    ''')

    with localconverter(ro.default_converter + pandas2ri.converter):
        df = ro.conversion.rpy2py(ro.globalenv['mast_result'])
    df = df.set_index("names")
    df = df.reindex(gene_names)
    result_dict = {
        "names": df.index.values,
        "logfoldchanges": df["logfoldchanges"].values,
        "scores": df["scores"].values,
        "pvals": df["pvals"].values,
        "pvals_adj": df["pvals_adj"].values
    }
    # Output format is exactly one result: key is the celltype name to be treated as "target"
    de_key = f"{group_1}_vs_{group_2}"
    adata.uns[key_added] = {de_key: result_dict}
    # Clean up R
    ro.r('rm(expr_mat, cell_data, mast_result); gc()')
    return adata.uns[key_added]

def main():
    parser = argparse.ArgumentParser(
        description='Perform differential expression analysis between two specified cell types'
    )
    parser.add_argument(
        '--config',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output',
        help='Directory to save output files (default: output)'
    )
    args = parser.parse_args()

    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    adata_path = config['adata_path']
    group_col = config['celltype_column']
    group_1, group_2 = config['celltype_values']

    print(f"Loading AnnData from {adata_path}...")
    adata = sc.read_h5ad(adata_path)

    # ---------------------- Basic filtering step on full matrix ----------------------
    print("Filtering cells with fewer than 200 genes expressed...")
    sc.pp.filter_cells(adata, min_genes=200)
    print("Filtering genes detected in fewer than 3 cells...")
    sc.pp.filter_genes(adata, min_cells=3)
    if 'n_genes' not in adata.obs:
        adata.obs['n_genes'] = (adata.X > 0).sum(axis=1).A1 if sp.issparse(adata.X) else (adata.X > 0).sum(axis=1)
    if 'n_counts' not in adata.obs:
        adata.obs['n_counts'] = adata.X.sum(axis=1).A1 if sp.issparse(adata.X) else adata.X.sum(axis=1)
    # -------------------------------------------------------------------------------

    # Ensure we have layers for DE (with all genes)
    if 'counts' not in adata.layers:
        adata.layers['counts'] = tocsr_if_not_sparse(adata.X)

    if 'norm_counts' not in adata.layers:
        print("Computing norm_counts layer (CPM/total count normalized, not log1p)...")
        temp = adata.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        adata.layers['norm_counts'] = tocsr_if_not_sparse(temp.X)
        del temp

    if 'log1p_norm' not in adata.layers:
        print("Computing log1p_norm layer (log1p of normalized counts)...")
        temp = adata.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        adata.layers['log1p_norm'] = tocsr_if_not_sparse(temp.X)
        del temp

    # Only retain the cells from the two specified groups
    mask = adata.obs[group_col].isin([group_1, group_2])
    adata = adata[mask].copy()
    print(f"Number of cells retained for DE: {adata.shape[0]}")

    # Re-categorize group_col so only two levels
    adata.obs[group_col] = pd.Categorical(
        adata.obs[group_col], categories=[group_1, group_2]
    )

    scanpy_wrapper = load_scanpy_wrapper()

    # Run DE using each method
    print("Running differential expression tests for the specified cell types...")

    # LN test (on norm_counts layer)
    ln_key = f"ln_{group_1}_vs_{group_2}"
    print(f"  Running LN test for {group_1} vs {group_2} ...")
    scanpy_wrapper.rank_genes_groups_ln(adata, group_col, 
        groups=[group_1],  # group 1 vs group 2 only
        reference=group_2,
        sparse=True,
        key_added=ln_key, 
        layer="norm_counts"
    )

    # Scanpy t-test (on log1p_norm layer)
    t_test_key = f"t_test_{group_1}_vs_{group_2}"
    print(f"  Running Scanpy t-test for {group_1} vs {group_2} ...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=group_col,
        groups=[group_1],
        reference=group_2,
        method="t-test",
        key_added=t_test_key,
        layer="log1p_norm",
        use_raw=False
    )

    # Scanpy wilcoxon (on log1p_norm layer)
    wilcoxon_key = f"wilcoxon_{group_1}_vs_{group_2}"
    print(f"  Running Scanpy wilcoxon test for {group_1} vs {group_2} ...")
    sc.tl.rank_genes_groups(
        adata,
        groupby=group_col,
        groups=[group_1],
        reference=group_2,
        method="wilcoxon",
        key_added=wilcoxon_key,
        layer="log1p_norm",
        use_raw=False
    )

    # Run MAST (on log1p_norm layer)
    mast_key = f"mast_{group_1}_vs_{group_2}"
    print(f"  Running MAST test for {group_1} vs {group_2} ...")
    run_mast_de(adata, group_col, group_1, group_2, log1p_layer="log1p_norm", key_added=mast_key)

    print("All DE tests complete.")

    # Write results to output file
    base_name = os.path.splitext(os.path.basename(adata_path))[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_{group_1}_vs_{group_2}_DE.h5ad")
    print(f"Saving updated AnnData with all DE results to {output_file} ...")
    adata.write(output_file)
    print("Done.")

if __name__ == '__main__':
    main()
