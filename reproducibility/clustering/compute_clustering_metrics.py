#!/usr/bin/env python3
"""
Script to compute clustering metrics (no DE performed in this script).

This script expects an AnnData file that already contains precomputed 
differential expression (DE) results for multiple clustering resolutions and DE methods.
It computes and saves cluster-level and cell-type level metrics (Jaccard indices, 
number of significant DE genes, average |LFC|, etc) as CSV files in an output folder.
Uses evaluation_utils for all metrics.

Usage:
    python compute_clustering_metrics.py --config config.yaml [--output-dir output/]
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import scanpy as sc

_rep_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _rep_dir not in sys.path:
    sys.path.insert(0, _rep_dir)
from evaluation_utils import (
    jaccard_index,
    recall,
    get_top_sig_genes_adata,
    get_all_sig_genes_adata,
    avg_abs_lfc_adata,
    avg_actual_abs_log2fc_adata,
    n_sig_genes_adata,
)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    required_fields = ["adata_path", "leiden_resolutions", "top_genes", "markers"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    if isinstance(config["leiden_resolutions"][0], list):
        config["leiden_resolutions"] = [tuple(r) for r in config["leiden_resolutions"]]
    return config

def main():
    parser = argparse.ArgumentParser(
        description='Compute clustering metrics using precomputed DE results'
    )
    parser.add_argument(
        '--config', type=str, required=True, help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--output-dir', type=str, default='output',
        help='Base output directory; metric CSVs are written to its "metrics" subfolder (default: output)'
    )
    parser.add_argument(
        '--skip-plots', action='store_true',
        help='(Ignored: plotting is always skipped in this version)'
    )

    args = parser.parse_args()
    config = load_config(args.config)
    # Metric CSVs live in the "metrics" subfolder of the output directory
    metrics_dir = os.path.join(args.output_dir, "metrics")
    os.makedirs(metrics_dir, exist_ok=True)

    adata_path = config['adata_path']
    leiden_resolutions = config['leiden_resolutions']
    top_genes = config['top_genes']
    markers = config['markers']

    resolution_keys = [r[1] for r in leiden_resolutions]
    cell_types = list(markers)
    cell_type_signatures = {ct: set(markers[ct]) for ct in cell_types}
    de_methods = ["ln", "t_test", "wilcoxon"]  # expects DE keys like ln_{resolution_key}, t_test_{resolution_key}, wilcoxon_{resolution_key}
    method_labels = {
        "ln": "LN test",
        "t_test": "Scanpy t-test (log1p)",
        "wilcoxon": "Scanpy Wilcoxon (log1p)"
    }

    print(f"Loading AnnData from {adata_path} ...")
    adata = sc.read_h5ad(adata_path)

    # ----------- Metrics computations start here ------------

    # 1. Get cluster signatures (top N sig genes) via evaluation_utils
    print("Preparing DE signatures ...")
    cluster_signatures = dict()
    for method in de_methods:
        cluster_signatures[method] = dict()
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            cluster_signatures[method][key] = dict()
            for cluster in clusters:
                cluster_signatures[method][key][cluster] = get_top_sig_genes_adata(
                    adata, cluster, top_n=top_genes, key=method_key, pval_threshold=0.05
                )

    # 2. All significant DE genes (padj < 0.05)
    all_sig_cluster_signatures = dict()
    for method in de_methods:
        all_sig_cluster_signatures[method] = dict()
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            all_sig_cluster_signatures[method][key] = dict()
            for cluster in clusters:
                all_sig_cluster_signatures[method][key][cluster] = get_all_sig_genes_adata(
                    adata, cluster, key=method_key, pval_threshold=0.05
                )

    # ---------- Signature-based metrics, computed for each signature set ----------
    # Two signature sets are evaluated: the top-N significant genes per cluster and
    # the full set of significant genes per cluster. Each produces its own Jaccard
    # and recall heatmap matrices plus avg-Jaccard, avg-recall and best-match CSVs,
    # tagged in the filename (e.g. "top20" vs "allsig").
    def compute_signature_metrics(signatures, tag):
        print(f"Computing Jaccard/recall heatmaps and summaries ({tag}) ...")
        jacc_records, recall_records, best_records = [], [], []
        for method in de_methods:
            for key in resolution_keys:
                clusters = sorted(
                    signatures[method][key].keys(),
                    key=lambda v: int(str(v)) if str(v).isdigit() else str(v),
                )
                jmat = np.zeros((len(clusters), len(cell_types)), dtype=float)
                rmat = np.zeros((len(clusters), len(cell_types)), dtype=float)
                for i, clus in enumerate(clusters):
                    sig_genes = signatures[method][key][clus]
                    for j, ct in enumerate(cell_types):
                        jmat[i, j] = jaccard_index(sig_genes, cell_type_signatures[ct])
                        rmat[i, j] = recall(sig_genes, cell_type_signatures[ct])
                pd.DataFrame(jmat, index=clusters, columns=cell_types).to_csv(
                    os.path.join(metrics_dir, f'jaccard_matrix_{tag}_{method}_{key}.csv')
                )
                pd.DataFrame(rmat, index=clusters, columns=cell_types).to_csv(
                    os.path.join(metrics_dir, f'recall_matrix_{tag}_{method}_{key}.csv')
                )
                for j, ct in enumerate(cell_types):
                    jcol, rcol = jmat[:, j], rmat[:, j]
                    common = {"resolution": key, "method": method_labels[method], "cell_type": ct}
                    jacc_records.append({**common, "avg_jaccard_sig": float(np.nanmean(jcol)) if jcol.size else np.nan})
                    recall_records.append({**common, "avg_recall_sig": float(np.nanmean(rcol)) if rcol.size else np.nan})
                    best_records.append({
                        **common,
                        "best_recall": float(rcol.max()) if rcol.size else np.nan,
                        "best_jaccard": float(jcol.max()) if jcol.size else np.nan,
                    })
        pd.DataFrame(jacc_records).to_csv(os.path.join(metrics_dir, f'avg_jaccard_{tag}_by_resolution.csv'), index=False)
        pd.DataFrame(recall_records).to_csv(os.path.join(metrics_dir, f'avg_recall_{tag}_by_resolution.csv'), index=False)
        pd.DataFrame(best_records).to_csv(os.path.join(metrics_dir, f'best_match_{tag}_by_resolution.csv'), index=False)

    compute_signature_metrics(cluster_signatures, f"top{top_genes}")
    compute_signature_metrics(all_sig_cluster_signatures, "allsig")

    # ---------- Metric 4: Average |LFC| (via evaluation_utils) ----------
    print("Computing average |LFC| (abs log fold change) per cluster ...")
    records = []
    for method in de_methods:
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            for clus in clusters:
                avg_abs_lfc = avg_abs_lfc_adata(adata, clus, key=method_key, pval_threshold=0.05)
                records.append({
                    "resolution": key,
                    "method": method_labels[method],
                    "cluster": clus,
                    "avg_abs_lfc_sig": avg_abs_lfc,
                })
                # For the t-test, also report the "actual" LFC it should use:
                # the log2 difference of mean log1p expression (group vs rest),
                # rather than scanpy's inflated expm1-back-transformed value.
                if method == "t_test":
                    actual_lfc = avg_actual_abs_log2fc_adata(
                        adata, clus, groupby=key, key=method_key,
                        layer="log1p_norm", pval_threshold=0.05,
                    )
                    records.append({
                        "resolution": key,
                        "method": "Scanpy t-test (actual LFC)",
                        "cluster": clus,
                        "avg_abs_lfc_sig": actual_lfc,
                    })
    df_lfc = pd.DataFrame(records)
    df_lfc.to_csv(
        os.path.join(metrics_dir, f'avg_lfc_by_resolution.csv'),
        index=False
    )

    # ---------- Metric 5: Number of significant DE genes (via evaluation_utils) ----------
    print("Computing number of significant DE genes per cluster ...")
    records = []
    for method in de_methods:
        for key in resolution_keys:
            method_key = f"{method}_{key}"
            clusters = [str(x) for x in sorted(adata.obs[key].unique(), key=lambda v: int(str(v)) if str(v).isdigit() else str(v))]
            for clus in clusters:
                n_sig = n_sig_genes_adata(adata, clus, key=method_key, pval_threshold=0.05)
                records.append({
                    "resolution": key,
                    "method": method_labels[method],
                    "cluster": clus,
                    "n_sig_genes": n_sig,
                })
    df_nsig = pd.DataFrame(records)
    df_nsig.to_csv(
        os.path.join(metrics_dir, f'n_sig_genes_by_resolution.csv'),
        index=False
    )

    print(f"\nAnalysis complete! Metric CSVs saved to {metrics_dir}/")


if __name__ == '__main__':
    main()

