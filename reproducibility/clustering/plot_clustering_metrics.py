#!/usr/bin/env python3
"""
Script to generate clustering metrics plots using precomputed metric CSV files.

Signature-based metrics are produced for two signature sets, distinguished by a
filename tag: the top-N significant genes per cluster ("top{N}") and the full set
of significant genes per cluster ("allsig"). For each tag this script expects:
    - avg_jaccard_{tag}_by_resolution.csv
    - avg_recall_{tag}_by_resolution.csv
    - best_match_{tag}_by_resolution.csv
    - jaccard_matrix_{tag}_{method}_{resolutionkey}.csv  (for heatmaps)
    - recall_matrix_{tag}_{method}_{resolutionkey}.csv    (for heatmaps)

Signature-set-independent metrics:
    - avg_lfc_by_resolution.csv
    - n_sig_genes_by_resolution.csv

Usage:
    python plot_clustering_metrics.py --metrics-dir output/metrics [--config config.yaml]

If config.yaml is provided, cell type marker order and method name mapping can be inferred for nicer plots.
"""

import argparse
import os
import yaml
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
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
        help='Directory containing metric CSV files (e.g. output/metrics)'
    )
    parser.add_argument(
        '--config', type=str, required=False,
        help='Optional YAML config file for cell type and resolution order'
    )
    parser.add_argument(
        '--output-dir', type=str, default=None,
        help='Directory to save plots (default: a sibling "figures" folder next to metrics-dir)'
    )
    parser.add_argument(
        '--skip-heatmaps', action='store_true',
        help='Skip plotting Jaccard/recall heatmaps (needs per-cluster CSVs)'
    )
    args = parser.parse_args()
    metrics_dir = args.metrics_dir
    # Default figures location: sibling "figures" folder next to the metrics dir
    default_fig_dir = os.path.join(os.path.dirname(os.path.normpath(metrics_dir)) or ".", "figures")
    output_dir = args.output_dir or default_fig_dir

    config = load_config(args.config) if args.config else {}
    cell_types = list(config.get('markers', {}).keys()) if 'markers' in config else None
    leiden_resolutions = config.get('leiden_resolutions', None)
    if leiden_resolutions:
        resolution_keys = [r[1] if isinstance(r, (list, tuple)) and len(r)==2 else str(r) for r in leiden_resolutions]
    else:
        resolution_keys = None
    top_genes = config.get('top_genes', 10)

    # Map raw method ids and labels to display labels (idempotent for labels).
    method_labels = {
        "LN test": "LN test",
        "Scanpy t-test (log1p)": "Scanpy t-test (log1p)",
        "Scanpy Wilcoxon (log1p)": "Scanpy Wilcoxon (log1p)",
        "ln": "LN test",
        "t_test": "Scanpy t-test (log1p)",
        "wilcoxon": "Scanpy Wilcoxon (log1p)",
    }

    os.makedirs(output_dir, exist_ok=True)
    sns.set(style="whitegrid")

    # Signature sets to plot: top-N significant genes and all significant genes.
    top_tag = f"top{top_genes}"
    sig_tags = [top_tag, "allsig"]
    tag_label = {top_tag: f"top {top_genes} sig genes", "allsig": "all significant genes"}

    def _read(name):
        path = os.path.join(metrics_dir, name)
        return pd.read_csv(path) if os.path.exists(path) else None

    # ---- Read signature-set-independent metric CSVs ----
    print("> Loading metric summary CSVs...")
    df_lfc = _read("avg_lfc_by_resolution.csv")
    df_nsig = _read("n_sig_genes_by_resolution.csv")

    # Derive method order from any available jaccard summary.
    ref = _read(f"avg_jaccard_{top_tag}_by_resolution.csv")
    if ref is not None and "method" in ref.columns:
        method_order = ref["method"].unique().tolist()
    else:
        method_order = ["LN test", "Scanpy t-test (log1p)", "Scanpy Wilcoxon (log1p)"]

    def _resolution_order(df):
        if resolution_keys:
            return resolution_keys
        vals = df["resolution"].astype(str).unique()
        return sorted(vals, key=lambda v: float(v) if str(v).replace('.', '', 1).isdigit() else str(v))

    resolution_order = resolution_keys
    if resolution_order is None:
        for df in (df_lfc, df_nsig, ref):
            if df is not None and "resolution" in df.columns:
                resolution_order = _resolution_order(df)
                break

    # Number of clusters per resolution (same across DE methods); derived from the
    # per-cluster n_sig CSV so we can annotate axis/subplot labels.
    n_clusters_by_res = {}
    if df_nsig is not None and {"resolution", "cluster"}.issubset(df_nsig.columns):
        n_clusters_by_res = (
            df_nsig.astype({"resolution": str})
            .groupby("resolution")["cluster"].nunique().to_dict()
        )

    _pretty_res = {
        "verylow": "Very low", "low": "Low", "mid": "Mid", "medium": "Medium",
        "high": "High", "veryhigh": "Very high",
    }

    def resolution_label(rk):
        """Map a resolution key (e.g. "leiden_verylow") to "Very low (2 clusters)"."""
        rk = str(rk)
        token = rk[len("leiden_"):] if rk.startswith("leiden_") else rk
        name = _pretty_res.get(token, token.replace("_", " ").capitalize())
        n = n_clusters_by_res.get(rk)
        return f"{name} ({n} clusters)" if n is not None else name

    def boxplot(df, ycol, ylabel, title, fname, legend_loc="upper right"):
        """Boxplot + stripplot of `ycol` grouped by resolution and method."""
        if df is None or ycol not in df.columns:
            return
        df = df.copy()
        df["resolution"] = df["resolution"].astype(str)
        # Hue order: configured methods first, then any extras (e.g. actual-LFC series).
        present = list(df["method"].unique())
        hue_order = [m for m in method_order if m in present]
        hue_order += [m for m in present if m not in hue_order]
        plt.figure(figsize=(10, 6))
        sns.boxplot(
            data=df, x="resolution", y=ycol, hue="method", showfliers=False,
            order=resolution_order, hue_order=hue_order,
        )
        sns.stripplot(
            data=df, x="resolution", y=ycol, hue="method", dodge=True, alpha=0.6,
            color='k', zorder=10, size=4, order=resolution_order, hue_order=hue_order,
        )
        ax = plt.gca()
        handles, labels = ax.get_legend_handles_labels()
        n = len(hue_order)
        plt.legend(handles[:n], labels[:n], title="Method", loc=legend_loc)
        ax.xaxis.set_major_locator(mticker.FixedLocator(ax.get_xticks()))
        ax.set_xticklabels([resolution_label(t.get_text()) for t in ax.get_xticklabels()],
                           rotation=20, ha="right")
        plt.ylabel(ylabel)
        plt.xlabel("Clustering resolution")
        plt.title(title)
        plt.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, fname), dpi=300, bbox_inches='tight')
        plt.close()

    # ---- Signature-set-independent boxplots ----
    print("> Plotting summary boxplots...")
    boxplot(
        df_lfc, "avg_abs_lfc_sig",
        "Average |LFC| of significant DE genes per cluster (FDR < 0.05)",
        "Average |LFC| of significant DE genes per cluster",
        "avg_lfc_by_resolution.png",
    )
    boxplot(
        df_nsig, "n_sig_genes",
        "Number of significant DE genes per cluster (FDR < 0.05)",
        "Number of significant DE genes per cluster",
        "n_sig_genes_by_resolution.png",
    )

    # ---- Signature-set-dependent boxplots, one set per tag ----
    for tag in sig_tags:
        lab = tag_label[tag]
        boxplot(
            _read(f"avg_jaccard_{tag}_by_resolution.csv"), "avg_jaccard_sig",
            "Average Jaccard index of cluster signatures to each cell type",
            f"Avg Jaccard ({lab})", f"avg_jaccard_{tag}_by_resolution.png",
        )
        boxplot(
            _read(f"avg_recall_{tag}_by_resolution.csv"), "avg_recall_sig",
            "Average recall of cell-type markers by cluster signatures",
            f"Avg recall of markers ({lab})", f"avg_recall_{tag}_by_resolution.png",
        )
        df_best = _read(f"best_match_{tag}_by_resolution.csv")
        boxplot(
            df_best, "best_recall",
            "Best-matching cluster recall of cell-type markers",
            f"Best-match recall of markers per cell type ({lab})",
            f"best_recall_{tag}_by_resolution.png", legend_loc="upper left",
        )
        boxplot(
            df_best, "best_jaccard",
            "Best-matching cluster Jaccard to cell-type markers",
            f"Best-match Jaccard per cell type ({lab})",
            f"best_jaccard_{tag}_by_resolution.png", legend_loc="upper left",
        )

    # ---- Per-method grouped heatmaps (Jaccard and recall), one figure per tag/method ----
    if not args.skip_heatmaps:
        from glob import glob
        from matplotlib.cm import ScalarMappable
        from matplotlib.colors import Normalize

        # Known DE methods, longest first so e.g. "t_test" is matched before "t".
        known_methods = sorted(["ln", "t_test", "wilcoxon"], key=len, reverse=True)

        def _parse_matrix_name(path, file_prefix):
            """Parse {file_prefix}_{tag}_{method}_{reskey}.csv into (tag, method, reskey).

            The tag (e.g. "top20"/"allsig") has no underscore, while both method
            (e.g. "t_test") and reskey (e.g. "leiden_mid") may contain "_", so we
            split off the tag first and then match a known method prefix.
            """
            base = os.path.basename(path).replace(".csv", "")
            rest = base[len(file_prefix) + 1:]  # {tag}_{method}_{reskey}
            tag, rest = rest.split("_", 1)
            method, reskey = None, rest
            for m in known_methods:
                if rest.startswith(m + "_"):
                    method, reskey = m, rest[len(m) + 1:]
                    break
            if method is None:  # fallback: split before the resolution key
                method, _, reskey = rest.partition("_leiden")
                reskey = "leiden" + reskey
            return tag, method, reskey

        def plot_grouped_heatmaps(file_prefix, out_prefix, cbar_label, metric_name, vmax_mode):
            """Build one figure per (tag, DE method), with a subplot per resolution.

            vmax_mode: "one" fixes the color scale to [0, 1]; "observed" uses the
            maximum value across that method's matrices.
            """
            files = glob(os.path.join(metrics_dir, f"{file_prefix}_*_*_*.csv"))
            if not files:
                print(f"  (No {file_prefix}_*.csv files found; skipping {metric_name} heatmaps.)")
                return
            groups = {}  # (tag, method) -> {reskey: path}
            for path in sorted(files):
                tag, method, reskey = _parse_matrix_name(path, file_prefix)
                groups.setdefault((tag, method), {})[reskey] = path

            for (tag, method), resmap in groups.items():
                if resolution_keys:
                    ordered = [rk for rk in resolution_keys if rk in resmap]
                    ordered += [rk for rk in resmap if rk not in ordered]
                else:
                    ordered = sorted(resmap)
                mats = {rk: pd.read_csv(resmap[rk], index_col=0) for rk in ordered}
                mlabel = method_labels.get(method, method)

                if vmax_mode == "observed":
                    observed = max((float(m.values.max()) for m in mats.values()), default=1.0)
                    vmax = observed if observed > 0 else 1.0
                else:
                    vmax = 1.0

                n = len(ordered)
                fig, axes = plt.subplots(1, n, figsize=(3.4 * n, 5.2), squeeze=False)
                axes = axes[0]
                for ax, rk in zip(axes, ordered):
                    mat = mats[rk]
                    sns.heatmap(
                        mat.values, ax=ax, annot=False, cmap="viridis",
                        vmin=0.0, vmax=vmax, cbar=False,
                        xticklabels=mat.columns, yticklabels=mat.index,
                    )
                    ax.set_title(resolution_label(rk))
                    ax.set_xlabel("Cell type marker")
                    ax.set_ylabel("Cluster")
                    plt.setp(ax.get_xticklabels(), rotation=90, fontsize=8)
                    plt.setp(ax.get_yticklabels(), fontsize=8)

                sm = ScalarMappable(cmap="viridis", norm=Normalize(vmin=0.0, vmax=vmax))
                fig.colorbar(sm, ax=list(axes), fraction=0.02, pad=0.02, label=cbar_label)
                taglab = tag_label.get(tag, tag)
                scale_note = f" (color scale 0–{vmax:.3g})" if vmax_mode == "observed" else ""
                fig.suptitle(
                    f"{mlabel}: {metric_name} of cluster signatures to cell-type markers "
                    f"— {taglab}{scale_note}",
                    fontsize=14,
                )
                savepath = os.path.join(output_dir, f"{out_prefix}_{tag}_{method}.png")
                fig.savefig(savepath, dpi=200, bbox_inches="tight")
                plt.close(fig)

        print("> Plotting Jaccard heatmaps (cluster x cell type), vmax = observed max ...")
        plot_grouped_heatmaps(
            "jaccard_matrix", "jaccard_heatmaps", "Jaccard Index", "Jaccard", vmax_mode="observed",
        )
        print("> Plotting recall heatmaps (cluster x cell type) ...")
        plot_grouped_heatmaps(
            "recall_matrix", "recall_heatmaps", "Recall (fraction of markers)", "Recall", vmax_mode="one",
        )

    print("\nPlots complete! Results saved to", output_dir)

if __name__ == '__main__':
    main()
