#!/usr/bin/env python3
"""
Script to analyze single-cell RNA-seq data across multiple Leiden resolutions.

This script performs differential expression analysis using both LN test and 
Scanpy's t-test (and wilcoxon) across multiple clustering resolutions, saving all
DE results to the updated AnnData file.

Usage:
    python run_multiresolution_de.py --config config.yaml [--output-dir output/]
"""

import argparse
import os
import sys
import tarfile
import yaml
import numpy as np
import scanpy as sc
import anndata as ad
import scipy.sparse as sp
import pandas as pd

# pandas >= 3.0 defaults to Arrow-backed string dtype, which anndata cannot
# serialize to h5ad. Disable it before any data is read so all string indices
# and columns stay as plain numpy object dtype.
try:
    pd.set_option("future.infer_string", False)
except (KeyError, ValueError):
    pass
from rpy2.robjects import pandas2ri, numpy2ri, r
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter

# Import scanpy_wrapper using the same pattern as the notebook
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
    required_fields = ['adata_path', 'leiden_resolutions']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Convert leiden_resolutions from list of lists to list of tuples
    if isinstance(config['leiden_resolutions'][0], list):
        config['leiden_resolutions'] = [tuple(r) for r in config['leiden_resolutions']]
    
    return config


def tocsr_if_not_sparse(x):
    """Convert ndarray/dense matrix to CSR sparse format if not already sparse."""
    if sp.issparse(x):
        return x.copy()
    return sp.csr_matrix(x)


def _resolve_10x_dir(path):
    """Return a directory containing a 10x matrix (matrix.mtx[.gz]).

    Accepts either a directory or a .tar.gz/.tgz archive of the 10x
    filtered_gene_bc_matrices. Archives are extracted next to the archive.
    """
    if os.path.isdir(path):
        search_root = path
    elif path.endswith((".tar.gz", ".tgz")):
        extract_root = os.path.join(
            os.path.dirname(os.path.abspath(path)),
            "_extracted_" + os.path.basename(path).split(".tar")[0],
        )
        with tarfile.open(path) as tf:
            tf.extractall(extract_root)
        search_root = extract_root
    else:
        raise ValueError(f"Unsupported 10x input path: {path}")

    for root, _dirs, files in os.walk(search_root):
        if any(f in files for f in ("matrix.mtx", "matrix.mtx.gz")):
            return root
    raise FileNotFoundError(f"No matrix.mtx found under {search_root}")


def load_input_adata(adata_path):
    """Load the input AnnData.

    If the path is an .h5ad it is read directly. Otherwise it is treated as raw
    10x data (a directory or .tar.gz archive) and the standard PBMC3k QC from the
    Seurat/Scanpy tutorial is applied, returning the full (all-genes) matrix.
    """
    if adata_path.endswith(".h5ad"):
        return sc.read_h5ad(adata_path)

    mtx_dir = _resolve_10x_dir(adata_path)
    print(f"Reading raw 10x matrix from {mtx_dir} ...")
    adata = sc.read_10x_mtx(mtx_dir, var_names="gene_symbols", cache=False)
    adata.var_names_make_unique()

    # Standard PBMC3k QC (matches the tutorial used to build pbmcs3k_pre.h5ad)
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    mito_genes = adata.var_names.str.startswith("MT-")
    counts_per_cell = np.asarray(adata.X.sum(axis=1)).ravel()
    mito_counts = np.asarray(adata[:, mito_genes].X.sum(axis=1)).ravel()
    adata.obs["percent_mito"] = mito_counts / counts_per_cell
    adata.obs["n_counts"] = counts_per_cell
    adata = adata[adata.obs["n_genes"] < 2500, :]
    adata = adata[adata.obs["percent_mito"] < 0.05, :]
    return adata.copy()

def coerce_arrow_strings(adata):
    """Coerce pandas Arrow-backed string arrays to plain numpy object dtype.

    anndata has no h5ad writer registered for ArrowStringArray, so writing an
    object whose obs/var (or raw.var) index or string columns use the pandas
    "string" dtype fails. Convert them to numpy object dtype before writing.
    """
    def _fix_df(df):
        # Force plain numpy object dtype; passing dtype=object explicitly prevents
        # pandas (>=3 / future.infer_string) from re-inferring an Arrow string dtype.
        df.index = pd.Index(
            np.asarray(df.index, dtype=object), dtype=object, name=df.index.name
        )
        for col in df.columns:
            if isinstance(df[col].dtype, pd.StringDtype) or str(df[col].dtype) in ("string", "str"):
                df[col] = pd.array(np.asarray(df[col], dtype=object), dtype=object)
        return df

    adata.obs = _fix_df(adata.obs)
    adata.var = _fix_df(adata.var)
    if adata.raw is not None:
        # adata.raw.var returns a fresh DataFrame, so mutating it in place does
        # not persist; rebuild raw from its own matrix with a cleaned var instead
        # (raw may hold all genes while adata.X is HVG-subset, so don't use adata).
        raw_var = _fix_df(adata.raw.var.copy())
        adata.raw = ad.AnnData(X=adata.raw.X, var=raw_var, obs=adata.obs)

def run_mast_de(
    adata,
    cluster_column,
    log1p_layer="log1p_norm",
    key_added="mast"
):
    """
    Run MAST differential expression analysis via rpy2 on a Scanpy AnnData object.

    The function uses the specified log1p-normalized data (adata.layers[log1p_layer]) 
    for DE, and writes the results to 
    adata.uns[f"{result_prefix}_{cluster_column}"], 
    as a Scanpy-format DE dictionary.

    Parameters
    ----------
    adata : AnnData
        The AnnData object containing single cell data.
    cluster_column : str
        Name of the column in adata.obs giving cluster labels.
    log1p_layer : str
        Layer in adata to use for log1p-normalized expression (default: "log1p_norm").
    key_added : str, optional
        Key to store the results in adata.uns. If None, defaults to "mast".
    """
    # Import R packages
    mast = importr('MAST')
    base = importr('base')
    stats = importr('stats')

    # Get normalized log-transformed expression matrix from specified layer
    if log1p_layer not in adata.layers:
        raise ValueError(f"log1p-normalized layer '{log1p_layer}' not found in adata.layers.")
    X = adata.layers[log1p_layer]
    if sp.issparse(X):
        X = X.toarray()
    # Ensure X is cells x genes
    cell_names = list(adata.obs_names)
    gene_names = list(adata.var_names)
    n_cells, n_genes = X.shape
    assert n_cells == len(cell_names), "Mismatch in number of cells"
    assert n_genes == len(gene_names), "Mismatch in number of genes"

    # Cluster information
    cluster_labels = adata.obs[cluster_column].values
    unique_clusters = sorted(pd.unique(cluster_labels))

    # Transfer expression matrix to R (genes x cells, Fortran order for R)
    X_T = np.asfortranarray(X.T)  # genes x cells in column-major for R
    r_expr = ro.r.matrix(
        ro.FloatVector(X_T.ravel(order='F')),
        nrow=n_genes, ncol=n_cells, byrow=False
    )
    r_expr.rownames = ro.StrVector(gene_names)
    r_expr.colnames = ro.StrVector(cell_names)
    ro.globalenv['expr_mat'] = r_expr
    del X, X_T

    # Results will be collected in Scanpy format: key is the cluster (as string) for one-vs-rest
    scanpy_format_de = {}
    # For each cluster, run one-vs-rest comparison
    for idx, target_cluster in enumerate(unique_clusters):
        print(f"      Running MAST for cluster '{target_cluster}' in '{cluster_column}' ({idx + 1}/{len(unique_clusters)}) ...")
        # Create binary grouping: target cluster vs rest
        binary_labels = np.where(cluster_labels == target_cluster, "target", "rest")
        # Create cell data frame
        cell_data = pd.DataFrame({
            'wellKey': cell_names,
            'group': binary_labels
        }, index=cell_names)
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_cell_data = ro.conversion.py2rpy(cell_data)
        ro.globalenv['cell_data'] = r_cell_data

        # Run MAST in R for this cluster
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
        # Ensure there are no duplicated gene names in adata.var
        df = df.set_index("names")
        # Reindex in adata.var order; fill NA for genes not tested
        df = df.reindex(gene_names)

        # Scanpy expects:
        #  'names' = gene names, 'scores' = test statistic (effect-signed), 'logfoldchanges', 
        #  'pvals', 'pvals_adj'
        result_dict = {
            "names": df.index.values,
            "logfoldchanges": df["logfoldchanges"].values,
            "scores": df["scores"].values,
            "pvals": df["pvals"].values,
            "pvals_adj": df["pvals_adj"].values
        }
        scanpy_format_de[str(target_cluster)] = result_dict

    # Clean up the shared R expression matrix
    ro.r('rm(expr_mat, cell_data, mast_result); gc()')

    # Write to AnnData in Scanpy format
    if "uns" not in dir(adata):
        adata.uns = {}
    adata.uns[key_added] = scanpy_format_de

    return scanpy_format_de

def main():
    parser = argparse.ArgumentParser(
        description='Analyze single-cell RNA-seq data across multiple Leiden resolutions'
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
    parser.add_argument(
        '--skip-mast',
        action='store_true',
        help='Skip the (slow) MAST DE test run via R'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}...")
    config = load_config(args.config)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Extract configuration parameters
    adata_path = config['adata_path']
    leiden_resolutions = config['leiden_resolutions']
    
    # Extract resolution values and keys
    resolution_values = [r[0] for r in leiden_resolutions]
    resolution_keys = [r[1] for r in leiden_resolutions]
    
    # Load AnnData (with all genes); supports raw 10x .tar.gz / directory input
    print(f"Loading input data from {adata_path}...")
    adata_full = load_input_adata(adata_path)

    # ---------------------- Basic filtering step on full matrix ----------------------
    print("Filtering cells with fewer than 200 genes expressed...")
    sc.pp.filter_cells(adata_full, min_genes=200)
    print("Filtering genes detected in fewer than 3 cells...")
    sc.pp.filter_genes(adata_full, min_cells=3)
    if 'n_genes' not in adata_full.obs:
        adata_full.obs['n_genes'] = (adata_full.X > 0).sum(axis=1).A1 if sp.issparse(adata_full.X) else (adata_full.X > 0).sum(axis=1)
    if 'n_counts' not in adata_full.obs:
        adata_full.obs['n_counts'] = adata_full.X.sum(axis=1).A1 if sp.issparse(adata_full.X) else adata_full.X.sum(axis=1)
    # -------------------------------------------------------------------------------

    # ---------------------- Prepare HVG-restricted AnnData for clustering -----------
    print("Finding top 2000 highly variable genes using seurat_v3 flavor (for clustering only)...")
    sc.pp.highly_variable_genes(
        adata_full,
        n_top_genes=2000,
        flavor='seurat_v3',
        subset=False,    # Don't subset in-place
        layer=None,
    )
    hvg_mask = adata_full.var['highly_variable'].values

    # Create a copy only for clustering with HVGs
    adata_hvg = adata_full[:, hvg_mask].copy()
    # NOTE: .X will be only HVGs, but obs is identical and any cell filtering is matched

    # Normalize and log1p (for clustering space)
    sc.pp.normalize_total(adata_hvg, target_sum=1e4)
    sc.pp.log1p(adata_hvg)
    sc.pp.scale(adata_hvg, max_value=10)

    # Compute PCA, neighbors, UMAP on HVG-restricted object
    if 'X_pca' not in adata_hvg.obsm:
        print("Computing PCA on HVGs...")
        sc.tl.pca(adata_hvg, svd_solver='arpack')
    if 'distances' not in adata_hvg.obsp:
        print("Computing neighbors on HVGs...")
        sc.pp.neighbors(adata_hvg, n_neighbors=10, n_pcs=40)
    if 'X_umap' not in adata_hvg.obsm:
        print("Computing UMAP on HVGs...")
        sc.tl.umap(adata_hvg)
    # -------------------------------------------------------------------------------

    # Compute Leiden clusters for each resolution
    print("Computing Leiden clusters for each resolution (using top 2000 HVG)...")
    for resolution_value, key_name in leiden_resolutions:
        if key_name not in adata_hvg.obs.columns:
            sc.tl.leiden(adata_hvg, key_added=key_name, resolution=resolution_value)
        else:
            print(f"  {key_name} already exists, skipping...")

    # Sort categories in all leiden resolutions by numerical order (modifies adata.obs in-place)
    for key_name in resolution_keys:
        adata_hvg.obs[key_name] = pd.Categorical(
            adata_hvg.obs[key_name],
            categories=sorted(adata_hvg.obs[key_name].unique()),
            ordered=True
        )

    # --- Transfer clustering/embedding annotations (obs, obsm, leiden assignments) to adata_full ---
    # Add cluster assignments and dimensionality reduction from adata_hvg to adata_full
    for key in resolution_keys:
        adata_full.obs[key] = adata_hvg.obs[key].copy()
    # Copy UMAP embedding (for convenience)
    if "X_umap" in adata_hvg.obsm:
        adata_full.obsm["X_umap"] = adata_hvg.obsm["X_umap"]
    # Copy PCA (optional)
    if "X_pca" in adata_hvg.obsm:
        adata_full.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    # Ensure adata_full.raw points to full matrix
    if adata_full.raw is None:
        adata_full.raw = adata_full

    # Ensure we have layers for DE (with all genes) in adata_full
    if 'counts' not in adata_full.layers:
        adata_full.layers['counts'] = tocsr_if_not_sparse(adata_full.X)

    if 'norm_counts' not in adata_full.layers:
        print("Computing norm_counts layer (CPM/total count normalized, not log1p)...")
        temp = adata_full.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        adata_full.layers['norm_counts'] = tocsr_if_not_sparse(temp.X)
        del temp

    if 'log1p_norm' not in adata_full.layers:
        print("Computing log1p_norm layer (log1p of normalized counts)...")
        temp = adata_full.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        adata_full.layers['log1p_norm'] = tocsr_if_not_sparse(temp.X)
        del temp

    # Load scanpy_wrapper
    scanpy_wrapper = load_scanpy_wrapper()
    
    # Run all DE tests once upfront for all resolutions (using ALL GENES! and transferred cluster assignments)
    print("Running differential expression tests for all resolutions (USE ALL GENES)...")
    for key_name in resolution_keys:
        # Print the number of clusters in each resolution before running DE
        n_clusters = adata_full.obs[key_name].nunique(dropna=True)
        print(f"  Processing {key_name}: {n_clusters} clusters")
        # Run LN test (on norm_counts layer, all genes)
        ln_key = f"ln_{key_name}"
        print(f"    Running LN test for '{key_name}' ...")
        scanpy_wrapper.rank_genes_groups_ln(adata_full, key_name, sparse=True, key_added=ln_key, layer="norm_counts")
        # Run Scanpy t-test (on log1p_norm layer, all genes)
        t_test_key = f"t_test_{key_name}"
        print(f"    Running Scanpy t-test for '{key_name}' ...")
        sc.tl.rank_genes_groups(adata_full, groupby=key_name, method="t-test", key_added=t_test_key, layer="log1p_norm", use_raw=False)
        # Run Scanpy wilcoxon (on log1p_norm layer, all genes)
        wilcoxon_key = f"wilcoxon_{key_name}"
        print(f"    Running Scanpy wilcoxon test for '{key_name}' ...")
        sc.tl.rank_genes_groups(adata_full, groupby=key_name, method="wilcoxon", key_added=wilcoxon_key, layer="log1p_norm", use_raw=False)
        # Run MAST (on log1p_norm layer, all genes)
        if not args.skip_mast:
            mast_key = f"mast_{key_name}"
            print(f"    Running MAST test for '{key_name}' ...")
            run_mast_de(adata_full, key_name, log1p_layer="log1p_norm", key_added=mast_key)
        else:
            print(f"    Skipping MAST test for '{key_name}' (--skip-mast).")
    
    print("All DE tests complete.")
    # Write updated AnnData with all DE results to file
    base_name = os.path.basename(adata_path.rstrip("/"))
    for ext in (".tar.gz", ".tgz", ".h5ad"):
        if base_name.endswith(ext):
            base_name = base_name[: -len(ext)]
            break
    else:
        base_name = os.path.splitext(base_name)[0]
    output_file = os.path.join(args.output_dir, f"{base_name}_DE.h5ad")
    print(f"Saving updated AnnData with all DE results to {output_file} ...")
    coerce_arrow_strings(adata_full)
    adata_full.write(output_file)
    
    print("Done.")

if __name__ == '__main__':
    main()
