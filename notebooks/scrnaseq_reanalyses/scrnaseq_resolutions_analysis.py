#!/usr/bin/env python3
"""
Script to analyze single-cell RNA-seq data across multiple Leiden resolutions.

This script performs differential expression analysis using both LN test and 
Scanpy's t-test across multiple clustering resolutions, and generates various
comparative plots and metrics.

Usage:
    python scrnaseq_resolutions_analysis.py --config config.yaml [--output-dir output/]
"""

import argparse
import os
import sys
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import entropy
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scanpy as sc
import importlib.util

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
    required_fields = ['adata_path', 'leiden_resolutions', 'top_genes', 'markers']
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    
    # Convert leiden_resolutions from list of lists to list of tuples
    if isinstance(config['leiden_resolutions'][0], list):
        config['leiden_resolutions'] = [tuple(r) for r in config['leiden_resolutions']]
    
    return config


def top_significant_genes(adata, group, top_genes=10, pval_threshold=0.05, key='rank_genes_groups'):
    """Get top significant genes for a group."""
    df = sc.get.rank_genes_groups_df(adata, group=group, key=key)
    sig_df = df[df['pvals_adj'] < pval_threshold]
    if sig_df.shape[0] == 0:
        return []
    sig_df = sig_df.sort_values('scores', ascending=False)
    return sig_df['names'].head(top_genes).tolist()

def compute_jaccard_index(signature_1, signature_2):
    """Compute the Jaccard index between two signatures."""
    return len(set(signature_1) & set(signature_2)) / len(set(signature_1) | set(signature_2))


def compute_jaccards(cluster_signature, cell_type_signatures, cell_types):
    """Return list of jaccard indices to each cell type."""
    jaccards = []
    for ct in cell_types:
        s = cell_type_signatures[ct]
        if len(cluster_signature | s) == 0:
            jaccard = np.nan
        else:
            jaccard = compute_jaccard_index(cluster_signature, s)
        jaccards.append(jaccard)
    return np.array(jaccards, dtype=float)


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
        '--skip-plots',
        action='store_true',
        help='Skip generating plots (only compute metrics)'
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
    top_genes = config['top_genes']
    markers = config['markers']
    
    # Extract resolution values and keys
    resolution_values = [r[0] for r in leiden_resolutions]
    resolution_keys = [r[1] for r in leiden_resolutions]
    
    # Load AnnData
    print(f"Loading AnnData from {adata_path}...")
    adata = sc.read_h5ad(adata_path)
    
    # Ensure we have normalized data
    if 'counts' not in adata.layers:
        adata.layers['counts'] = adata.X.copy()
    if adata.raw is None:
        adata.raw = adata
    
    # Create normalized copy if needed
    adata_norm = adata.copy()
    if 'log1p' not in str(type(adata.X)):
        sc.pp.normalize_total(adata_norm, target_sum=1e4)
    
    # Normalize and log-transform for scanpy analysis
    if 'log1p' not in str(type(adata.X)):
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.pp.scale(adata, max_value=10)
    
    # Compute PCA and neighbors if not already done
    if 'X_pca' not in adata.obsm:
        print("Computing PCA...")
        sc.tl.pca(adata, svd_solver='arpack')
    
    if 'distances' not in adata.obsp:
        print("Computing neighbors...")
        sc.pp.neighbors(adata, n_neighbors=10, n_pcs=40)
    
    # Compute UMAP if not already done
    if 'X_umap' not in adata.obsm:
        print("Computing UMAP...")
        sc.tl.umap(adata)
    
    # Create leiden clusters for each resolution
    print("Computing Leiden clusters for each resolution...")
    for resolution_value, key_name in leiden_resolutions:
        if key_name not in adata.obs.columns:
            sc.tl.leiden(adata, key_added=key_name, resolution=resolution_value)
            adata_norm.obs[key_name] = adata.obs[key_name]
        else:
            print(f"  {key_name} already exists, skipping...")
    
    # Sort categories in all leiden resolutions by numerical order
    import pandas as pd
    for key_name in resolution_keys:
        adata.obs[key_name] = pd.Categorical(
            adata.obs[key_name],
            categories=sorted(adata.obs[key_name].unique()),
            ordered=True
        )
        adata_norm.obs[key_name] = adata.obs[key_name]
    
    # Load scanpy_wrapper
    scanpy_wrapper = load_scanpy_wrapper()
    
    # Run all DE tests once upfront for all resolutions
    print("Running differential expression tests for all resolutions...")
    for key_name in resolution_keys:
        print(f"  Processing {key_name}...")
        # Run LN test (use unique key for each resolution to avoid overwriting)
        ln_key = f"ln_{key_name}"
        scanpy_wrapper.rank_genes_groups_ln(adata_norm, key_name, sparse=True, key_added=ln_key)
        # Run Scanpy t-test (use unique key for each resolution to avoid overwriting)
        t_test_key = f"t_test_{key_name}"
        sc.tl.rank_genes_groups(adata, groupby=key_name, method="t-test", key_added=t_test_key)
    
    print("DE tests complete. Computing signatures and generating plots...")
    
    # Compute signatures for each resolution (using pre-computed results)
    print("Computing LN signatures...")
    ln_signatures = {}
    for key_name in resolution_keys:
        clusters = adata_norm.obs[key_name].unique()
        clusters = np.array(sorted(clusters.astype(int))).astype(str)
        ln_key = f"ln_{key_name}"
        ln_signatures[key_name] = {
            c: top_significant_genes(adata_norm, c, top_genes=top_genes, key=ln_key)
            for c in clusters
        }
    
    print("Computing log1p signatures...")
    log1p_signatures = {}
    for key_name in resolution_keys:
        clusters = adata.obs[key_name].unique()
        clusters = np.array(sorted(clusters.astype(int))).astype(str)
        t_test_key = f"t_test_{key_name}"
        log1p_signatures[key_name] = {
            c: top_significant_genes(adata, c, top_genes=top_genes, key=t_test_key)
            for c in clusters
        }
    
    # Get cell types from markers
    cell_types = list(markers.keys())
    
    if not args.skip_plots:
        # Plot 1: Jaccard index heatmaps
        print("Generating Jaccard index heatmaps...")
        ln_mats = {}
        for key_name in resolution_keys:
            clusters = sorted(ln_signatures[key_name].keys())
            ln_mats[key_name] = np.zeros((len(clusters), len(cell_types)))
            for i, c in enumerate(clusters):
                for j, c2 in enumerate(cell_types):
                    ln_mats[key_name][i, j] = compute_jaccard_index(
                        ln_signatures[key_name][c], markers[c2]
                    )
        
        log1p_mats = {}
        for key_name in resolution_keys:
            clusters = sorted(log1p_signatures[key_name].keys())
            log1p_mats[key_name] = np.zeros((len(clusters), len(cell_types)))
            for i, c in enumerate(clusters):
                for j, c2 in enumerate(cell_types):
                    log1p_mats[key_name][i, j] = compute_jaccard_index(
                        log1p_signatures[key_name][c], markers[c2]
                    )
        
        # Get vmin/vmax for shared color scale
        all_mats = list(ln_mats.values()) + list(log1p_mats.values())
        vmin = min(mat.min() for mat in all_mats)
        vmax = max(mat.max() for mat in all_mats)
        
        # Plotting
        n_resolutions = len(resolution_keys)
        fig, axes = plt.subplots(
            2, n_resolutions,
            figsize=(7*n_resolutions, 10),
            constrained_layout=True,
            sharex='col',
            sharey=False
        )
        
        # Top row: LN
        for col, key_name in enumerate(resolution_keys):
            clusters = sorted(ln_signatures[key_name].keys())
            axes[0, col].pcolormesh(ln_mats[key_name], vmin=vmin, vmax=vmax)
            axes[0, col].set_title(f'LN {key_name}')
            axes[0, col].set_ylabel(f'{key_name} cluster')
            axes[0, col].set_yticks(np.arange(len(clusters)) + 0.5)
            axes[0, col].set_yticklabels(clusters)
            axes[0, col].set_xticklabels([])
        
        # Bottom row: log1p
        last_c = None
        for col, key_name in enumerate(resolution_keys):
            clusters = sorted(log1p_signatures[key_name].keys())
            last_c = axes[1, col].pcolormesh(log1p_mats[key_name], vmin=vmin, vmax=vmax)
            axes[1, col].set_title(f'log1p {key_name}')
            axes[1, col].set_ylabel(f'{key_name} cluster')
            axes[1, col].set_xlabel('Cell type')
            axes[1, col].set_yticks(np.arange(len(clusters)) + 0.5)
            axes[1, col].set_yticklabels(clusters)
            axes[1, col].set_xticks(np.arange(len(cell_types)) + 0.5)
            axes[1, col].set_xticklabels(cell_types, rotation=90)
        
        # Shared colorbar
        if last_c is not None:
            fig.colorbar(
                last_c, ax=axes, orientation='vertical',
                fraction=0.03, pad=0.02, label='Jaccard Index'
            )
        
        plt.savefig(
            os.path.join(args.output_dir, 'jaccard_heatmaps.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        print(f"  Saved to {os.path.join(args.output_dir, 'jaccard_heatmaps.png')}")
        
        # Plot 2: Average Jaccard index across resolutions
        print("Generating average Jaccard index plot...")
        cell_type_signatures = {ct: set(markers[ct]) for ct in cell_types}
        top_n = top_genes
        
        records = []
        for key in resolution_keys:
            clusters = sorted([str(x) for x in adata_norm.obs[key].unique()])
            # Use pre-computed DE results (no need to rerun)
            
            ln_cluster_signatures = []
            ttest_cluster_signatures = []
            ln_key = f"ln_{key}"
            for group in clusters:
                df_ln = sc.get.rank_genes_groups_df(adata_norm, group=group, key=ln_key)
                sig_ln = df_ln[df_ln["pvals_adj"] < 0.05].sort_values("scores", ascending=False).head(top_n)
                ln_cluster_signatures.append(set(sig_ln["names"]))
                df_t = sc.get.rank_genes_groups_df(adata, group=group, key=f"t_test_{key}")
                sig_t = df_t[df_t["pvals_adj"] < 0.05].sort_values("scores", ascending=False).head(top_n)
                ttest_cluster_signatures.append(set(sig_t["names"]))
            
            for ct in cell_types:
                s = cell_type_signatures[ct]
                # LN test
                ln_jaccards = []
                for cluster_signature in ln_cluster_signatures:
                    if len(cluster_signature | s) == 0:
                        jaccard = np.nan
                    else:
                        jaccard = len(cluster_signature & s) / len(cluster_signature | s)
                    ln_jaccards.append(jaccard)
                if len(ln_jaccards) > 0:
                    ln_mean_jaccard = np.nanmean(ln_jaccards)
                else:
                    ln_mean_jaccard = np.nan
                records.append({
                    "resolution": key,
                    "method": "LN test",
                    "cell_type": ct,
                    "avg_jaccard_sig": ln_mean_jaccard,
                })
                # Scanpy t-test
                t_jaccards = []
                for cluster_signature_t in ttest_cluster_signatures:
                    if len(cluster_signature_t | s) == 0:
                        jaccard = np.nan
                    else:
                        jaccard = len(cluster_signature_t & s) / len(cluster_signature_t | s)
                    t_jaccards.append(jaccard)
                if len(t_jaccards) > 0:
                    t_mean_jaccard = np.nanmean(t_jaccards)
                else:
                    t_mean_jaccard = np.nan
                records.append({
                    "resolution": key,
                    "method": "Scanpy t-test (log1p)",
                    "cell_type": ct,
                    "avg_jaccard_sig": t_mean_jaccard,
                })
        
        df_plot = pd.DataFrame(records)
        df_plot.to_csv(
            os.path.join(args.output_dir, 'avg_jaccard_by_resolution.csv'),
            index=False
        )
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_plot,
            x="resolution",
            y="avg_jaccard_sig",
            hue="method",
            width=0.6,
            showfliers=False,
            gap=0.1,
        )
        sns.stripplot(
            data=df_plot,
            x="resolution",
            y="avg_jaccard_sig",
            hue="method",
            dodge=True,
            alpha=0.6,
            color='k',
            zorder=10,
            size=4,
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        methods = df_plot["method"].unique()
        n_methods = len(methods)
        plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
        plt.ylabel("Average Jaccard index of cluster signatures to each cell type")
        plt.xlabel("Clustering resolution")
        plt.title(f"Average Jaccard index across cluster signatures to each cell type (top {top_genes} significant genes)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, 'avg_jaccard_by_resolution.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        print(f"  Saved to {os.path.join(args.output_dir, 'avg_jaccard_by_resolution.png')}")
        
        # Plot 2b: Average Jaccard index using ALL significant genes (not just top N)
        print("Generating average Jaccard index plot (all significant genes)...")
        records = []
        for key in resolution_keys:
            clusters = sorted([str(x) for x in adata_norm.obs[key].unique()])
            # Use pre-computed DE results (no need to rerun)
            
            ln_cluster_signatures = []
            ttest_cluster_signatures = []
            ln_key = f"ln_{key}"
            for group in clusters:
                df_ln = sc.get.rank_genes_groups_df(adata_norm, group=group, key=ln_key)
                # Use ALL significant genes, not just top N
                sig_ln = df_ln[df_ln["pvals_adj"] < 0.05].sort_values("scores", ascending=False)
                ln_cluster_signatures.append(set(sig_ln["names"]))
                df_t = sc.get.rank_genes_groups_df(adata, group=group, key=f"t_test_{key}")
                # Use ALL significant genes, not just top N
                sig_t = df_t[df_t["pvals_adj"] < 0.05].sort_values("scores", ascending=False)
                ttest_cluster_signatures.append(set(sig_t["names"]))
            
            for ct in cell_types:
                s = cell_type_signatures[ct]
                # LN test
                ln_jaccards = []
                for cluster_signature in ln_cluster_signatures:
                    jaccard = compute_jaccard_index(cluster_signature, s)
                    ln_jaccards.append(jaccard)
                if len(ln_jaccards) > 0:
                    ln_mean_jaccard = np.nanmean(ln_jaccards)
                else:
                    ln_mean_jaccard = np.nan
                records.append({
                    "resolution": key,
                    "method": "LN test",
                    "cell_type": ct,
                    "avg_jaccard_sig": ln_mean_jaccard,
                })
                # Scanpy t-test
                t_jaccards = []
                for cluster_signature_t in ttest_cluster_signatures:
                    jaccard = compute_jaccard_index(cluster_signature_t, s)
                    t_jaccards.append(jaccard)
                if len(t_jaccards) > 0:
                    t_mean_jaccard = np.nanmean(t_jaccards)
                else:
                    t_mean_jaccard = np.nan
                records.append({
                    "resolution": key,
                    "method": "Scanpy t-test (log1p)",
                    "cell_type": ct,
                    "avg_jaccard_sig": t_mean_jaccard,
                })
        
        df_plot = pd.DataFrame(records)
        df_plot.to_csv(
            os.path.join(args.output_dir, 'avg_jaccard_by_resolution_all_genes.csv'),
            index=False
        )
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df_plot,
            x="resolution",
            y="avg_jaccard_sig",
            hue="method",
            width=0.6,
            showfliers=False,
            gap=0.1,
        )
        sns.stripplot(
            data=df_plot,
            x="resolution",
            y="avg_jaccard_sig",
            hue="method",
            dodge=True,
            alpha=0.6,
            color='k',
            zorder=10,
            size=4,
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        methods = df_plot["method"].unique()
        n_methods = len(methods)
        plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
        plt.ylabel("Average Jaccard index of cluster signatures to each cell type")
        plt.xlabel("Clustering resolution")
        plt.title("Average Jaccard index across cluster signatures to each cell type (all significant genes)")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, 'avg_jaccard_by_resolution_all_genes.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        print(f"  Saved to {os.path.join(args.output_dir, 'avg_jaccard_by_resolution_all_genes.png')}")
        
        # Plot 4: Average LFC
        print("Generating average LFC plot...")
        records = []
        for key in resolution_keys:
            # Use pre-computed DE results (no need to rerun)
            clusters = sorted([str(x) for x in adata_norm.obs[key].unique()])
            ln_key = f"ln_{key}"
            for group in clusters:
                # LN test
                df_ln = sc.get.rank_genes_groups_df(adata_norm, group=group, key=ln_key)
                sig_ln = df_ln[df_ln["pvals_adj"] < 0.05]
                if len(sig_ln) > 0:
                    avg_lfc_ln = sig_ln["logfoldchanges"].abs().mean()
                else:
                    avg_lfc_ln = np.nan
                records.append({
                    "resolution": key,
                    "method": "LN test",
                    "avg_abs_lfc_sig": avg_lfc_ln,
                    "cluster": group,
                })
                # t-test (Scanpy)
                df_t = sc.get.rank_genes_groups_df(adata, group=group, key=f"t_test_{key}")
                sig_t = df_t[df_t["pvals_adj"] < 0.05]
                if len(sig_t) > 0:
                    avg_lfc_t = sig_t["logfoldchanges"].abs().mean()
                else:
                    avg_lfc_t = np.nan
                records.append({
                    "resolution": key,
                    "method": "Scanpy t-test (log1p)",
                    "avg_abs_lfc_sig": avg_lfc_t,
                    "cluster": group,
                })
        
        df_plot = pd.DataFrame(records)
        df_plot.to_csv(
            os.path.join(args.output_dir, 'avg_lfc_by_resolution.csv'),
            index=False
        )
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df_plot,
            x="resolution",
            y="avg_abs_lfc_sig",
            hue="method",
            width=0.6,
            showfliers=False,
            gap=0.1,
        )
        sns.stripplot(
            data=df_plot,
            x="resolution",
            y="avg_abs_lfc_sig",
            hue="method",
            dodge=True,
            alpha=0.6,
            color='k',
            zorder=10,
            size=4,
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        methods = df_plot["method"].unique()
        n_methods = len(methods)
        plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
        plt.ylabel("Average |LFC| of significant DE genes per cluster (FDR < 0.05)")
        plt.xlabel("Clustering resolution")
        plt.title("Average |LFC| of significant DE genes per cluster across methods and resolutions")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, 'avg_lfc_by_resolution.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        print(f"  Saved to {os.path.join(args.output_dir, 'avg_lfc_by_resolution.png')}")
        
        # Plot 5: Number of significant genes
        print("Generating number of significant genes plot...")
        records = []
        for key in resolution_keys:
            # Use pre-computed DE results (no need to rerun)
            clusters = sorted([str(x) for x in adata_norm.obs[key].unique()])
            ln_key = f"ln_{key}"
            for group in clusters:
                # LN test
                df_ln = sc.get.rank_genes_groups_df(adata_norm, group=group, key=ln_key)
                n_sig_ln = (df_ln["pvals_adj"] < 0.05).sum()
                records.append({
                    "resolution": key,
                    "method": "LN test",
                    "n_sig_genes": n_sig_ln,
                    "cluster": group,
                })
                # t-test (Scanpy)
                df_t = sc.get.rank_genes_groups_df(adata, group=group, key=f"t_test_{key}")
                n_sig_t = (df_t["pvals_adj"] < 0.05).sum()
                records.append({
                    "resolution": key,
                    "method": "Scanpy t-test (log1p)",
                    "n_sig_genes": n_sig_t,
                    "cluster": group,
                })
        
        df_plot = pd.DataFrame(records)
        df_plot.to_csv(
            os.path.join(args.output_dir, 'n_sig_genes_by_resolution.csv'),
            index=False
        )
        
        plt.figure(figsize=(8, 5))
        sns.boxplot(
            data=df_plot,
            x="resolution",
            y="n_sig_genes",
            hue="method",
            width=0.6,
            showfliers=False,
            gap=0.1,
        )
        sns.stripplot(
            data=df_plot,
            x="resolution",
            y="n_sig_genes",
            hue="method",
            dodge=True,
            alpha=0.6,
            color='k',
            zorder=10,
            size=4,
        )
        handles, labels = plt.gca().get_legend_handles_labels()
        methods = df_plot["method"].unique()
        n_methods = len(methods)
        plt.legend(handles[:n_methods], labels[:n_methods], title="Method", loc="upper right")
        plt.ylabel("Number of significant DE genes per cluster (FDR < 0.05)")
        plt.xlabel("Clustering resolution")
        plt.title("Number of significant DE genes per cluster across methods and resolutions")
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.output_dir, 'n_sig_genes_by_resolution.png'),
            dpi=300, bbox_inches='tight'
        )
        plt.close()
        print(f"  Saved to {os.path.join(args.output_dir, 'n_sig_genes_by_resolution.png')}")
    
    print(f"\nAnalysis complete! Results saved to {args.output_dir}/")


if __name__ == '__main__':
    main()
