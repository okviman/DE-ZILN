#!/usr/bin/env python3
"""
Script to compute clustering metrics (no DE performed in this script).

This script expects an AnnData file that already contains precomputed 
differential expression (DE) results for multiple clustering resolutions and DE methods.
It computes and saves cluster-level and cell-type level metrics (Jaccard indices, 
number of significant DE genes, average |LFC|, etc) as CSV files in an output folder.

Usage:
    python compute_clustering_metrics.py --config config.yaml [--output-dir output/]
"""

import argparse
import os
import yaml
import numpy as np
import pandas as pd
import scanpy as sc

def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    required_fields = ['adata_path', 'leiden_resolutions', 'top_genes', 'markers']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    # Convert leiden_resolutions from list of lists to list of tuples
    if isinstance(config['leiden_resolutions'][0], list):
        config['leiden_resolutions'] = [tuple(r) for r in config['leiden_resolutions']]
    return config

def compute_jaccard_index(set1, set2):
    set1 = set(set1)
    set2 = set(set2)
    denom = len(set1 | set2)
    if denom == 0:
        return np.nan
    return len(set1 & set2) / denom

def top_significant_genes(adata, group, top_genes=10, pval_threshold=0.05, key='rank_genes_groups'):
    """Get top significant genes for a group."""
    df = sc.get.rank_genes_groups_df(adata, group=group, key=key)
    sig_df = df[df['pvals_adj'] < pval_threshold]
    if sig_df.shape[0] == 0:
        return []
    sig_df = sig_df.sort_values('scores', ascending=False)
    return sig_df['names'].head(top_genes).tolist()

def main():
    parser = argparse.ArgumentParser(
        description='Compute clustering metrics using precomputed DE results'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='Directory to save output files (default: output)'
    )
    parser.add_argument(
        '--skip-plots', action='store_true',
        help='(Ignored: plotting is always skipped in this version)'
    )

    args = parser.parse_args()
    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)

    adata_path = config['adata_path']
    leiden_resolutions = config['leiden_resolutions']
    top_genes = config['top_genes']
    markers = config['markers']

    resolution_keys = [r[1] for r in leiden_resolutions]
    cell_types = list(markers)
    cell_type_signatures = {ct: set(markers[ct]) for ct in cell_types}
    de_methods = ["ln", "t_test"]  # expects DE keys like ln_{resolution_key} and t_test_{resolution_key}
    method_labels = {
        "ln": "LN test",
        "t_test": "Scanpy t-test (log1p)"
    }

    print(f"Loading AnnData from {adata_path} ...")
    adata = sc.read_h5ad(adata_path)

    # ----------- Metrics computations start here ------------

    # 1. Get cluster signatures (for each method, for each resolution)
    #    Signatures = top N significant DE genes (by score, padj < 0.05)
    print("Preparing DE signatures ...")
    cluster_signatures = dict()  # cluster_signatures[method][resolution][cluster] = set of genes
    for method in de_methods:
        cluster_signatures[method] = dict()
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            cluster_signatures[method][key] = dict()
            for cluster in clusters:
                genes = top_significant_genes(
                    adata, cluster, top_genes=top_genes, key=method_key
                )
                cluster_signatures[method][key][cluster] = set(genes)

    # 2. Get cluster signatures using all significant DE genes (padj < 0.05, any score)
    all_sig_cluster_signatures = dict()
    for method in de_methods:
        all_sig_cluster_signatures[method] = dict()
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            all_sig_cluster_signatures[method][key] = dict()
            for cluster in clusters:
                df = sc.get.rank_genes_groups_df(adata, group=cluster, key=method_key)
                sig_genes = df[df["pvals_adj"] < 0.05]["names"].tolist()
                all_sig_cluster_signatures[method][key][cluster] = set(sig_genes)

    # ---------- Metric 1: Jaccard (top N sig DE genes) ----------
    print("Computing Jaccard index heatmaps (top N sig DE genes) ...")
    # Jaccard matrix per method and resolution: clusters x cell_types
    jaccard_matrices = dict()  # jaccard_matrices[method][resolution] = matrix
    for method in de_methods:
        jaccard_matrices[method] = {}
        for key in resolution_keys:
            clusters = sorted(cluster_signatures[method][key].keys(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))
            mat = np.zeros((len(clusters), len(cell_types)), dtype=float)
            for i, clus in enumerate(clusters):
                sig_genes = cluster_signatures[method][key][clus]
                for j, ct in enumerate(cell_types):
                    mat[i, j] = compute_jaccard_index(sig_genes, cell_type_signatures[ct])
            jaccard_matrices[method][key] = (mat, clusters)
            df = pd.DataFrame(mat, index=clusters, columns=cell_types)
            df.to_csv(
                os.path.join(args.output_dir, f'jaccard_matrix_top{top_genes}_{method}_{key}.csv')
            )

    # ---------- Metric 2: Average Jaccard index (top N) per cluster/celltype/resolution ----------
    print("Computing average Jaccard index (top N sig DE genes) ...")
    records = []
    for method in de_methods:
        for key in resolution_keys:
            clusters = sorted(cluster_signatures[method][key].keys(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))
            for ct in cell_types:
                cluster_jaccards = []
                s = cell_type_signatures[ct]
                for clus in clusters:
                    sig_genes = cluster_signatures[method][key][clus]
                    jacc = compute_jaccard_index(sig_genes, s)
                    cluster_jaccards.append(jacc)
                records.append({
                    'resolution': key,
                    'method': method_labels[method],
                    'cell_type': ct,
                    'avg_jaccard_sig': np.nanmean(cluster_jaccards) if cluster_jaccards else np.nan,
                })
    df_jaccard = pd.DataFrame(records)
    df_jaccard.to_csv(
        os.path.join(args.output_dir, f'avg_jaccard_by_resolution.csv'),
        index=False
    )

    # ---------- Metric 3: Average Jaccard index (ALL sig DE genes) -----------
    print("Computing average Jaccard index (all significant DE genes) ...")
    records = []
    for method in de_methods:
        for key in resolution_keys:
            clusters = sorted(all_sig_cluster_signatures[method][key].keys(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))
            for ct in cell_types:
                cluster_jaccards = []
                s = cell_type_signatures[ct]
                for clus in clusters:
                    sig_genes = all_sig_cluster_signatures[method][key][clus]
                    jacc = compute_jaccard_index(sig_genes, s)
                    cluster_jaccards.append(jacc)
                records.append({
                    'resolution': key,
                    'method': method_labels[method],
                    'cell_type': ct,
                    'avg_jaccard_sig': np.nanmean(cluster_jaccards) if cluster_jaccards else np.nan,
                })
    df_jaccard_all = pd.DataFrame(records)
    df_jaccard_all.to_csv(
        os.path.join(args.output_dir, f'avg_jaccard_by_resolution_all_genes.csv'),
        index=False
    )

    # ---------- Metric 4: Average |LFC| of significant DE genes per cluster ----------
    print("Computing average |LFC| (abs log fold change) per cluster ...")
    records = []
    for method in de_methods:
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            for clus in clusters:
                df = sc.get.rank_genes_groups_df(adata, group=clus, key=method_key)
                sig_df = df[df["pvals_adj"] < 0.05]
                avg_abs_lfc = sig_df["logfoldchanges"].abs().mean() if not sig_df.empty else np.nan
                records.append({
                    "resolution": key,
                    "method": method_labels[method],
                    "cluster": clus,
                    "avg_abs_lfc_sig": avg_abs_lfc,
                })
    df_lfc = pd.DataFrame(records)
    df_lfc.to_csv(
        os.path.join(args.output_dir, f'avg_lfc_by_resolution.csv'),
        index=False
    )

    # ---------- Metric 5: Number of significant DE genes per cluster ----------
    print("Computing number of significant DE genes per cluster ...")
    records = []
    for method in de_methods:
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            for clus in clusters:
                df = sc.get.rank_genes_groups_df(adata, group=clus, key=method_key)
                n_sig = (df["pvals_adj"] < 0.05).sum()
                records.append({
                    "resolution": key,
                    "method": method_labels[method],
                    "cluster": clus,
                    "n_sig_genes": n_sig,
                })
    df_nsig = pd.DataFrame(records)
    df_nsig.to_csv(
        os.path.join(args.output_dir, f'n_sig_genes_by_resolution.csv'),
        index=False
    )

    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()

