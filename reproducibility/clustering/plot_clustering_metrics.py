#!/usr/bin/env python3
"""
Script to generate clustering metrics plots using precomputed metric CSV files.

This script expects CSV files for metrics such as:
    - avg_jaccard_by_resolution.csv
    - avg_jaccard_by_resolution_all_genes.csv
    - avg_lfc_by_resolution.csv
    - n_sig_genes_by_resolution.csv
    - (optionally, jaccard_matrix_top{N}_{method}_{resolutionkey}.csv files for heatmaps)

Usage:
    python plot_clustering_metrics.py --metrics-dir output/ [--config config.yaml]

If config.yaml is provided, cell type marker order and method name mapping can be inferred for nicer plots.
"""

import argparse
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def load_config(config_path):
    """Load configuration from YAML file, if present."""
    if config_path is None:
        return {}
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def main():
    parser = argparse.ArgumentParser(
        description='Generate clustering metrics plots from CSV summaries'
    )
    parser.add_argument(
        '--metrics-dir', type=str, required=True,
        help='Directory containing metric CSV files (from compute script)'
    )
    parser.add_argument(
        '--config', type=str, required=False,
        help='Optional YAML config file for cell type and resolution order'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save plots (default: metrics-dir)'
    )
    parser.add_argument(
        '--skip-heatmaps', action='store_true',
        help='Skip plotting Jaccard heatmaps (needs per-cluster CSVs)'
    )
    args = parser.parse_args()
    metrics_dir = args.metrics_dir
    output_dir = args.output_dir or metrics_dir

    config = load_config(args.config) if args.config else {}
    cell_types = list(config.get('markers', {}).keys()) if 'markers' in config else None
    leiden_resolutions = config.get('leiden_resolutions', None)
    if leiden_resolutions:
        resolution_keys = [r[1] if isinstance(r, (list, tuple)) and len(r)==2 else str(r) for r in leiden_resolutions]
    else:
        resolution_keys = None
    top_genes = config.get('top_genes', 10)

    # In absence of config, just infer method-labels from the metrics file present.
    method_labels = {
        "LN test": "LN test", "Scanpy t-test (log1p)": "Scanpy t-test (log1p)",  # fallback
        "ln": "LN test",
        "t_test": "Scanpy t-test (log1p)"
    }

    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # ---- Read metrics CSVs ----
    print("> Loading metric summary CSVs...")
    df_jaccard = pd.read_csv(os.path.join(metrics_dir, "avg_jaccard_by_resolution.csv"))
    df_jaccard_all = pd.read_csv(os.path.join(metrics_dir, "avg_jaccard_by_resolution_all_genes.csv"))
    df_lfc = pd.read_csv(os.path.join(metrics_dir, "avg_lfc_by_resolution.csv"))
    df_nsig = pd.read_csv(os.path.join(metrics_dir, "n_sig_genes_by_resolution.csv"))

    # Try to sort method labels if present
    if "method" in df_jaccard.columns:
        unique_methods = df_jaccard["method"].unique().tolist()
    else:
        unique_methods = ["LN test", "Scanpy t-test (log1p)"]

    # Try to get resolution column as str
    for df in [df_jaccard, df_jaccard_all, df_lfc, df_nsig]:
        if "resolution" in df.columns:
            df["resolution"] = df["resolution"].astype(str)

    # Use order from config or as in data
    method_order = unique_methods
    resolution_order = resolution_keys if resolution_keys else sorted(df_jaccard["resolution"].unique(), key=lambda v: float(v) if str(v).replace('.', '', 1).isdigit() else str(v))

    # --- Boxplots (Avg Jaccard, All sig, |LFC|, N sig genes) ---
    print("> Plotting summary boxplots...")
    # Average Jaccard (top N sig genes)
    plt.figure(figsize=(10,6))
    sns.boxplot(
        data=df_jaccard,
        x="resolution", y="avg_jaccard_sig", hue="method", showfliers=False,
        order=resolution_order, hue_order=method_order
    )
    sns.stripplot(
        data=df_jaccard,
        x="resolution", y="avg_jaccard_sig", hue="method", dodge=True, alpha=0.6, color='k', zorder=10, size=4,
        order=resolution_order, hue_order=method_order
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    n_methods = len(method_order)
    plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
    plt.ylabel("Average Jaccard index of cluster signatures to each cell type")
    plt.xlabel("Clustering resolution")
    plt.title(f"Avg Jaccard (top {top_genes} sig genes)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_jaccard_by_resolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Average Jaccard (ALL sig DE genes)
    plt.figure(figsize=(10,6))
    sns.boxplot(
        data=df_jaccard_all,
        x="resolution", y="avg_jaccard_sig", hue="method", showfliers=False,
        order=resolution_order, hue_order=method_order
    )
    sns.stripplot(
        data=df_jaccard_all,
        x="resolution", y="avg_jaccard_sig", hue="method", dodge=True, alpha=0.6, color='k', zorder=10, size=4,
        order=resolution_order, hue_order=method_order
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    n_methods = len(method_order)
    plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
    plt.ylabel("Average Jaccard index of cluster signatures to each cell type")
    plt.xlabel("Clustering resolution")
    plt.title(f"Avg Jaccard (all significant genes)")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_jaccard_by_resolution_all_genes.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Average |LFC| (abs log-fold-change)
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df_lfc,
        x="resolution", y="avg_abs_lfc_sig", hue="method", showfliers=False,
        order=resolution_order, hue_order=method_order
    )
    sns.stripplot(
        data=df_lfc,
        x="resolution", y="avg_abs_lfc_sig", hue="method", dodge=True, alpha=0.6, color='k', zorder=10, size=4,
        order=resolution_order, hue_order=method_order
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    n_methods = len(method_order)
    plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
    plt.ylabel("Average |LFC| of significant DE genes per cluster (FDR < 0.05)")
    plt.xlabel("Clustering resolution")
    plt.title("Average |LFC| of significant DE genes per cluster")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'avg_lfc_by_resolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Number of significant DE genes per cluster
    plt.figure(figsize=(8,5))
    sns.boxplot(
        data=df_nsig,
        x="resolution", y="n_sig_genes", hue="method", showfliers=False,
        order=resolution_order, hue_order=method_order
    )
    sns.stripplot(
        data=df_nsig,
        x="resolution", y="n_sig_genes", hue="method", dodge=True, alpha=0.6, color='k', zorder=10, size=4,
        order=resolution_order, hue_order=method_order
    )
    handles, labels = plt.gca().get_legend_handles_labels()
    n_methods = len(method_order)
    plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
    plt.ylabel("Number of significant DE genes per cluster (FDR < 0.05)")
    plt.xlabel("Clustering resolution")
    plt.title("Number of significant DE genes per cluster")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'n_sig_genes_by_resolution.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # ---- Plot Jaccard index heatmaps for clusters:matrix ----
    # Try to infer jaccard_matrix_topN_{method}_{reskey}.csv format (for each method, res)
    if not args.skip_heatmaps:
        print("> Plotting Jaccard heatmaps (cluster x cell type) if matrix CSVs are present...")
        from glob import glob
        # Try to infer methods and resolutions from CSVs present
        jaccard_files = glob(os.path.join(metrics_dir, "jaccard_matrix_top*_*_*.csv"))
        if len(jaccard_files) == 0:
            print("  (No jaccard_matrix_topN_{method}_{reskey}.csv files found; skipping heatmaps.)")
        else:
            import numpy as np
            for path in sorted(jaccard_files):
                base = os.path.basename(path).replace(".csv","")
                # e.g. jaccard_matrix_top10_ln_0.8
                parts = base.split("_")
                try:  # robust to clustering string name
                    topN = int(parts[3].replace('top','')) if parts[3].startswith("top") else int(parts[2].replace('top',''))
                    method = parts[-2]
                    reskey = parts[-1]
                except Exception:
                    method = parts[-2]
                    reskey = parts[-1]
                    topN = top_genes
                mat = pd.read_csv(path, index_col=0)
                plt.figure(figsize=(1.5+0.3*len(mat.columns), 1.5+0.25*len(mat)))
                ax = sns.heatmap(mat.values, annot=True, fmt=".2f",
                                 xticklabels=mat.columns, yticklabels=mat.index,
                                 cmap="viridis", vmin=0.0, vmax=1.0, cbar_kws={'label': "Jaccard Index"})
                ax.set_xlabel("Cell type marker signature")
                ax.set_ylabel(f"{reskey} cluster")
                mlabel = method_labels.get(method, method)
                plt.title(f"Jaccard index: {mlabel} - {reskey} (top {topN})")
                plt.tight_layout()
                savepath = os.path.join(output_dir, f'{base}.png')
                plt.savefig(savepath, dpi=200, bbox_inches='tight')
                plt.close()

    print("\nPlots complete! Results saved to", output_dir)

if __name__ == '__main__':
    main()
