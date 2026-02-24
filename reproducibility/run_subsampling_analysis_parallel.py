#!/usr/bin/env python3
"""
Parallelized sub-sampling effect analysis using shared memory.

Uses de_utils for DE (LN, t-test, wilcoxon, MAST) and evaluation_utils for
all metrics (Jaccard, AUPRC, precision@k, avg |LFC|, n_sig_genes).
Input adata.X = raw counts (spot-level); after subsampling we aggregate to
cell-level counts and run the same analysis as the celltype pipeline.

Usage:
    python run_subsampling_analysis_parallel.py --config config.yaml [--output-dir output/] [--n-workers 8]
"""

import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)

# Force single-threaded BLAS/OpenMP BEFORE any numeric library imports
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["R_DATATABLE_NUM_THREADS"] = "1"
os.environ["BLIS_NUM_THREADS"] = "1"

import argparse
import multiprocessing
import time
import warnings

try:
    from multiprocessing import shared_memory
    USE_SHARED_MEMORY = True
except ImportError:
    shared_memory = None
    USE_SHARED_MEMORY = False  # Python < 3.8: workers load adata from path

import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
import yaml

from de_utils import (
    prepare_adata_layers,
    run_ln_de,
    run_ttest_de,
    run_wilcoxon_de,
    run_mast_de,
)
from evaluation_utils import (
    jaccard_top_adata,
    jaccard_all_sig_adata,
    aupr_adata,
    precision_at_k_adata,
    avg_abs_lfc_adata,
    n_sig_genes_adata,
    n_genes_de_table_adata,
    frac_de_genes_valid_score_adata,
    jaccard_top_df,
    jaccard_all_sig_df,
    aupr_df,
    precision_at_k_df,
    avg_abs_lfc_df,
    n_sig_genes_df,
    n_genes_de_table_df,
    frac_de_genes_valid_score_df,
    gsea_nes_adata,
    gsea_nes_df,
)

warnings.filterwarnings("ignore")

K_VALUES = [10, 20, 50, 100]
PVAL_THRESHOLD = 0.05


def _plot_box_pandas(df, x_col, y_col, hue_col, ax=None):
    """Boxplot via pandas/matplotlib to avoid seaborn UnboundLocalError in some versions."""
    if ax is None:
        ax = plt.gca()
    x_vals = sorted(df[x_col].dropna().unique(), key=str)
    hue_vals = sorted(df[hue_col].dropna().unique(), key=str)
    n_hue = len(hue_vals)
    width = 0.7 / max(n_hue, 1)
    for xi, x in enumerate(x_vals):
        for hi, h in enumerate(hue_vals):
            vals = df.loc[(df[x_col] == x) & (df[hue_col] == h), y_col].dropna()
            if len(vals) == 0:
                continue
            pos = xi + (hi - (n_hue - 1) / 2) * width * 1.1
            bp = ax.boxplot(
                [vals.values], positions=[pos], widths=width * 0.9,
                patch_artist=True, showfliers=False,
            )
            for box in bp["boxes"]:
                box.set_facecolor(plt.cm.tab10(hi % 10))
    ax.set_xticks(range(len(x_vals)))
    ax.set_xticklabels([str(x) for x in x_vals])
    from matplotlib.patches import Patch
    handles = [Patch(facecolor=plt.cm.tab10(i % 10)) for i in range(len(hue_vals))]
    ax.legend(handles=handles, labels=hue_vals, title=hue_col)
    return ax


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    required = [
        "adata_path",
        "subsampling_fractions",
        "n_replicates",
        "top_genes",
        "cluster_column",
        "cell_id_column",
        "markers",
    ]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field in config: {field}")
    return config


# ---- Shared memory and worker ----

_shared_data = {}


def _limit_threads():
    """Apply single-thread limits for BLAS/OpenMP and R."""
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(limits=1)
    except ImportError:
        pass
    try:
        import rpy2.robjects as ro
        ro.r("invisible(suppressMessages({if(requireNamespace(\"data.table\",quietly=TRUE)) data.table::setDTthreads(1)}))")
        ro.r("invisible(suppressMessages({if(requireNamespace(\"RhpcBLASctl\",quietly=TRUE)) {RhpcBLASctl::blas_set_num_threads(1); RhpcBLASctl::omp_set_num_threads(1)}}))")
    except Exception:
        pass


def init_worker(
    shm_data_name,
    shm_indices_name,
    shm_indptr_name,
    shm_cell_ids_name,
    shm_clusters_name,
    data_shape,
    data_dtype,
    indices_shape,
    indices_dtype,
    indptr_shape,
    indptr_dtype,
    cell_ids_shape,
    clusters_shape,
    matrix_shape,
    var_names,
    cluster_column,
    cell_id_column,
    markers,
    top_genes,
    skip_mast=False,
    skip_gsea=False,
    order_only=False,
):
    """Initialize worker with shared memory references (Python 3.8+)."""
    global _shared_data
    _limit_threads()
    _shared_data["shm_data"] = shared_memory.SharedMemory(name=shm_data_name)
    _shared_data["shm_indices"] = shared_memory.SharedMemory(name=shm_indices_name)
    _shared_data["shm_indptr"] = shared_memory.SharedMemory(name=shm_indptr_name)
    _shared_data["shm_cell_ids"] = shared_memory.SharedMemory(name=shm_cell_ids_name)
    _shared_data["shm_clusters"] = shared_memory.SharedMemory(name=shm_clusters_name)
    _shared_data["data"] = np.ndarray(data_shape, dtype=data_dtype, buffer=_shared_data["shm_data"].buf)
    _shared_data["indices"] = np.ndarray(indices_shape, dtype=indices_dtype, buffer=_shared_data["shm_indices"].buf)
    _shared_data["indptr"] = np.ndarray(indptr_shape, dtype=indptr_dtype, buffer=_shared_data["shm_indptr"].buf)
    _shared_data["cell_ids"] = np.ndarray(cell_ids_shape, dtype="<U20", buffer=_shared_data["shm_cell_ids"].buf)
    _shared_data["clusters"] = np.ndarray(clusters_shape, dtype="<U20", buffer=_shared_data["shm_clusters"].buf)
    _shared_data["matrix_shape"] = matrix_shape
    _shared_data["var_names"] = var_names
    _shared_data["cluster_column"] = cluster_column
    _shared_data["cell_id_column"] = cell_id_column
    _shared_data["markers"] = markers
    _shared_data["top_genes"] = top_genes
    _shared_data["skip_mast"] = skip_mast
    _shared_data["skip_gsea"] = skip_gsea
    _shared_data["order_only"] = order_only


def init_worker_load_from_path(adata_path, cluster_column, cell_id_column, markers, top_genes, skip_mast=False, skip_gsea=False, order_only=False):
    """Initialize worker by loading adata from disk (fallback when shared_memory not available, e.g. Python 3.7)."""
    global _shared_data
    _limit_threads()
    adata = sc.read_h5ad(adata_path)
    if sparse.issparse(adata.X):
        X_csr = adata.X.tocsr()
    else:
        X_csr = sparse.csr_matrix(adata.X)
    _shared_data["data"] = np.array(X_csr.data, dtype=np.float32)
    _shared_data["indices"] = np.array(X_csr.indices, dtype=np.int32)
    _shared_data["indptr"] = np.array(X_csr.indptr, dtype=np.int64)
    max_len = 20
    cell_ids_arr = adata.obs[cell_id_column].astype(str).values
    clusters_arr = adata.obs[cluster_column].astype(str).values
    _shared_data["cell_ids"] = np.array([s[:max_len].ljust(max_len) for s in cell_ids_arr], dtype=f"<U{max_len}")
    _shared_data["clusters"] = np.array([s[:max_len].ljust(max_len) for s in clusters_arr], dtype=f"<U{max_len}")
    _shared_data["matrix_shape"] = X_csr.shape
    _shared_data["var_names"] = list(adata.var_names)
    _shared_data["cluster_column"] = cluster_column
    _shared_data["cell_id_column"] = cell_id_column
    _shared_data["markers"] = markers
    _shared_data["top_genes"] = top_genes
    _shared_data["skip_mast"] = skip_mast
    _shared_data["skip_gsea"] = skip_gsea
    _shared_data["order_only"] = order_only


def process_task(task):
    """Process a single (fraction, replicate) task: subsample -> aggregate -> DE (de_utils) -> metrics (evaluation_utils)."""
    t0 = time.time()
    fraction, replicate = task
    global _shared_data

    X = sparse.csr_matrix(
        (_shared_data["data"], _shared_data["indices"], _shared_data["indptr"]),
        shape=_shared_data["matrix_shape"],
    )
    cell_ids = _shared_data["cell_ids"]
    clusters = _shared_data["clusters"]
    var_names = _shared_data["var_names"]
    cluster_column = _shared_data["cluster_column"]
    cell_id_column = _shared_data["cell_id_column"]
    markers = _shared_data["markers"]
    top_genes = _shared_data["top_genes"]
    skip_mast = _shared_data.get("skip_mast", False)
    skip_gsea = _shared_data.get("skip_gsea", False)
    order_only = _shared_data.get("order_only", False)

    np.random.seed(replicate)
    unique_cells = np.unique(cell_ids)
    n_cells = len(unique_cells)
    n_genes = X.shape[1]
    cell_to_cluster = {cell_ids[i]: clusters[i] for i in range(len(cell_ids))}

    aggregated_counts = np.zeros((n_cells, n_genes), dtype=np.float32)
    cell_order = []
    cluster_assignments = []
    for i, cell_id in enumerate(unique_cells):
        spot_indices = np.where(cell_ids == cell_id)[0]
        n_spots = len(spot_indices)
        n_select = max(1, int(np.round(n_spots * fraction)))
        if n_select >= n_spots:
            selected_indices = spot_indices
        else:
            selected_indices = np.random.choice(spot_indices, size=n_select, replace=False)
        cell_counts = np.asarray(X[selected_indices, :].sum(axis=0)).flatten()
        aggregated_counts[i, :] = cell_counts
        cell_order.append(cell_id)
        cluster_assignments.append(cell_to_cluster[cell_id])

    adata_agg = sc.AnnData(
        X=sparse.csr_matrix(aggregated_counts),
        obs=pd.DataFrame(
            {cell_id_column: cell_order, cluster_column: cluster_assignments},
            index=[str(c) for c in cell_order],
        ),
        var=pd.DataFrame(index=var_names),
    )
    adata_agg.obs[cluster_column] = pd.Categorical(adata_agg.obs[cluster_column])

    # DE via de_utils (same as celltype: layers + LN, t-test, wilcoxon, MAST) with per-method timing
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_layers = time.time()
        prepare_adata_layers(adata_agg)
        time_layers_s = time.time() - t_layers
        t_ln = time.time()
        run_ln_de(adata_agg, cluster_column, key_added="ln_test", layer="norm_counts", sparse=True)
        time_ln_s = time.time() - t_ln
        t_ttest = time.time()
        run_ttest_de(adata_agg, cluster_column, key_added="t_test", layer="log1p_norm", use_raw=False)
        time_ttest_s = time.time() - t_ttest
        t_wilcoxon = time.time()
        run_wilcoxon_de(adata_agg, cluster_column, key_added="wilcoxon", layer="log1p_norm", use_raw=False)
        time_wilcoxon_s = time.time() - t_wilcoxon
        t_mast = time.time()
        if skip_mast:
            mast_results = {}
        else:
            try:
                mast_results = run_mast_de(adata_agg, cluster_column, group_1=None, group_2=None, log1p_layer="log1p_norm")
            except Exception:
                mast_results = {}
        time_mast_s = time.time() - t_mast

    # Metrics via evaluation_utils. Markers are per cluster: each cluster evaluated only vs its own ground-truth set.
    cluster_list = sorted(adata_agg.obs[cluster_column].unique())
    records = []

    for cluster in cluster_list:
        cluster_key = str(cluster).strip()
        marker_set = set(markers.get(cluster_key, []))
        if not marker_set:
            continue
        # LN test
        avg_lfc_ln = avg_abs_lfc_adata(adata_agg, cluster, key="ln_test", pval_threshold=PVAL_THRESHOLD)
        n_sig_ln = n_sig_genes_adata(adata_agg, cluster, key="ln_test", pval_threshold=PVAL_THRESHOLD, positive_score_only=True)
        jt = jaccard_top_adata(adata_agg, cluster, marker_set, top_genes, key="ln_test", pval_threshold=PVAL_THRESHOLD)
        ja = jaccard_all_sig_adata(adata_agg, cluster, marker_set, key="ln_test", pval_threshold=PVAL_THRESHOLD, positive_score_only=True)
        au = aupr_adata(adata_agg, cluster, marker_set, key="ln_test", order_only=order_only)
        pk = precision_at_k_adata(adata_agg, cluster, marker_set, K_VALUES, key="ln_test")
        if skip_gsea:
            gsea_ln = np.nan
        else:
            try:
                gsea_ln = gsea_nes_adata(adata_agg, cluster, marker_set, key="ln_test", order_only=order_only)
            except Exception:
                gsea_ln = np.nan
        n_genes_ln = n_genes_de_table_adata(adata_agg, cluster, key="ln_test")
        frac_valid_ln = frac_de_genes_valid_score_adata(adata_agg, cluster, key="ln_test")
        records.append({
            "cluster": cluster, "method": "LN test",
            "jaccard_top": jt, "jaccard_all": ja, "auprc": au,
            "precision_at_10": pk.get(10, np.nan), "precision_at_20": pk.get(20, np.nan),
            "precision_at_50": pk.get(50, np.nan), "precision_at_100": pk.get(100, np.nan),
            "avg_abs_lfc": avg_lfc_ln, "n_sig_genes": n_sig_ln, "gsea_nes": gsea_ln,
            "n_genes_de_table": n_genes_ln, "frac_de_genes_valid_score": frac_valid_ln,
            "de_time_s": time_ln_s,
            "fraction": fraction, "replicate": replicate,
        })
        # t-test
        avg_lfc_tt = avg_abs_lfc_adata(adata_agg, cluster, key="t_test", pval_threshold=PVAL_THRESHOLD)
        n_sig_tt = n_sig_genes_adata(adata_agg, cluster, key="t_test", pval_threshold=PVAL_THRESHOLD, positive_score_only=True)
        jt = jaccard_top_adata(adata_agg, cluster, marker_set, top_genes, key="t_test", pval_threshold=PVAL_THRESHOLD)
        ja = jaccard_all_sig_adata(adata_agg, cluster, marker_set, key="t_test", pval_threshold=PVAL_THRESHOLD, positive_score_only=True)
        au = aupr_adata(adata_agg, cluster, marker_set, key="t_test", order_only=order_only)
        pk = precision_at_k_adata(adata_agg, cluster, marker_set, K_VALUES, key="t_test")
        if skip_gsea:
            gsea_tt = np.nan
        else:
            try:
                gsea_tt = gsea_nes_adata(adata_agg, cluster, marker_set, key="t_test", order_only=order_only)
            except Exception:
                gsea_tt = np.nan
        n_genes_tt = n_genes_de_table_adata(adata_agg, cluster, key="t_test")
        frac_valid_tt = frac_de_genes_valid_score_adata(adata_agg, cluster, key="t_test")
        records.append({
            "cluster": cluster, "method": "t-test",
            "jaccard_top": jt, "jaccard_all": ja, "auprc": au,
            "precision_at_10": pk.get(10, np.nan), "precision_at_20": pk.get(20, np.nan),
            "precision_at_50": pk.get(50, np.nan), "precision_at_100": pk.get(100, np.nan),
            "avg_abs_lfc": avg_lfc_tt, "n_sig_genes": n_sig_tt, "gsea_nes": gsea_tt,
            "n_genes_de_table": n_genes_tt, "frac_de_genes_valid_score": frac_valid_tt,
            "de_time_s": time_ttest_s,
            "fraction": fraction, "replicate": replicate,
        })
        # Wilcoxon
        avg_lfc_w = avg_abs_lfc_adata(adata_agg, cluster, key="wilcoxon", pval_threshold=PVAL_THRESHOLD)
        n_sig_w = n_sig_genes_adata(adata_agg, cluster, key="wilcoxon", pval_threshold=PVAL_THRESHOLD, positive_score_only=True)
        jt = jaccard_top_adata(adata_agg, cluster, marker_set, top_genes, key="wilcoxon", pval_threshold=PVAL_THRESHOLD)
        ja = jaccard_all_sig_adata(adata_agg, cluster, marker_set, key="wilcoxon", pval_threshold=PVAL_THRESHOLD, positive_score_only=True)
        au = aupr_adata(adata_agg, cluster, marker_set, key="wilcoxon", order_only=order_only)
        pk = precision_at_k_adata(adata_agg, cluster, marker_set, K_VALUES, key="wilcoxon")
        try:
            gsea_w = gsea_nes_adata(adata_agg, cluster, marker_set, key="wilcoxon", order_only=order_only)
        except Exception:
            gsea_w = np.nan
        n_genes_w = n_genes_de_table_adata(adata_agg, cluster, key="wilcoxon")
        frac_valid_w = frac_de_genes_valid_score_adata(adata_agg, cluster, key="wilcoxon")
        records.append({
            "cluster": cluster, "method": "Wilcoxon",
            "jaccard_top": jt, "jaccard_all": ja, "auprc": au,
            "precision_at_10": pk.get(10, np.nan), "precision_at_20": pk.get(20, np.nan),
            "precision_at_50": pk.get(50, np.nan), "precision_at_100": pk.get(100, np.nan),
            "avg_abs_lfc": avg_lfc_w, "n_sig_genes": n_sig_w, "gsea_nes": gsea_w,
            "n_genes_de_table": n_genes_w, "frac_de_genes_valid_score": frac_valid_w,
            "de_time_s": time_wilcoxon_s,
            "fraction": fraction, "replicate": replicate,
        })
        # MAST (skip appending when --skip-mast)
        if not skip_mast:
            mast_df = mast_results.get(str(cluster), None)
            avg_lfc_mast = avg_abs_lfc_df(mast_df, pval_threshold=PVAL_THRESHOLD)
            n_sig_mast = n_sig_genes_df(mast_df, pval_threshold=PVAL_THRESHOLD, positive_score_only=True) if mast_df is not None and len(mast_df) > 0 else 0
            jt = jaccard_top_df(mast_df, marker_set, top_genes, pval_threshold=PVAL_THRESHOLD) if mast_df is not None and len(mast_df) > 0 else np.nan
            ja = jaccard_all_sig_df(mast_df, marker_set, pval_threshold=PVAL_THRESHOLD, positive_score_only=True) if mast_df is not None and len(mast_df) > 0 else np.nan
            au = aupr_df(mast_df, marker_set, order_only=order_only) if mast_df is not None and len(mast_df) > 0 else np.nan
            pk = precision_at_k_df(mast_df, marker_set, K_VALUES) if mast_df is not None and len(mast_df) > 0 else {k: np.nan for k in K_VALUES}
            try:
                gsea_mast = gsea_nes_df(mast_df, marker_set, order_only=order_only) if mast_df is not None and len(mast_df) > 0 else np.nan
            except Exception:
                gsea_mast = np.nan
            n_genes_mast = n_genes_de_table_df(mast_df) if mast_df is not None and len(mast_df) > 0 else 0
            frac_valid_mast = frac_de_genes_valid_score_df(mast_df) if mast_df is not None and len(mast_df) > 0 else np.nan
            records.append({
                "cluster": cluster,
                "method": "MAST",
                "jaccard_top": jt,
                "jaccard_all": ja,
                "auprc": au,
                "precision_at_10": pk.get(10, np.nan),
                "precision_at_20": pk.get(20, np.nan),
                "precision_at_50": pk.get(50, np.nan),
                "precision_at_100": pk.get(100, np.nan),
                "avg_abs_lfc": avg_lfc_mast,
                "n_sig_genes": n_sig_mast,
                "gsea_nes": gsea_mast,
                "n_genes_de_table": n_genes_mast,
                "frac_de_genes_valid_score": frac_valid_mast,
                "de_time_s": time_mast_s,
                "fraction": fraction,
                "replicate": replicate,
            })

    task_time = time.time() - t0
    return records, task_time


def main():
    parser = argparse.ArgumentParser(description="Parallelized sub-sampling analysis (DE via de_utils, metrics via evaluation_utils)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="output_subsampling_parallel", help="Output directory")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--skip-mast", action="store_true", help="Skip MAST DE (faster runs for debugging)")
    parser.add_argument("--skip-gsea", action="store_true", help="Skip GSEA NES (avoids gseapy errors/noise)")
    parser.add_argument("--order-only", action="store_true", help="Use only rank order for AUPR/GSEA (replace scores with ranks)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PARALLELIZED SUB-SAMPLING ANALYSIS (reproducibility)")
    print("=" * 80)
    print(f"Using {args.n_workers} workers, DE and metrics from de_utils / evaluation_utils")

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"\n[1/7] Config loaded. Fractions: {config['subsampling_fractions']}, Replicates: {config['n_replicates']}")
    if args.skip_mast:
        print("  --skip-mast: MAST DE disabled (faster run)")
    if args.order_only:
        print("  --order-only: AUPR/GSEA use rank order only (ignore score magnitude)")
    if not USE_SHARED_MEMORY:
        print("  (Python < 3.8: each worker will load adata from disk; no shared memory)")

    tasks = []
    for fraction in config["subsampling_fractions"]:
        for rep in range(config["n_replicates"]):
            tasks.append((fraction, rep))

    shm_handles = []
    if USE_SHARED_MEMORY:
        print(f"\n[2/7] Loading adata from {config['adata_path']}...")
        adata = sc.read_h5ad(config["adata_path"])
        if sparse.issparse(adata.X):
            X_csr = adata.X.tocsr()
        else:
            X_csr = sparse.csr_matrix(adata.X)
        data_arr = np.array(X_csr.data, dtype=np.float32)
        indices_arr = np.array(X_csr.indices, dtype=np.int32)
        indptr_arr = np.array(X_csr.indptr, dtype=np.int64)
        cell_ids_arr = adata.obs[config["cell_id_column"]].astype(str).values
        clusters_arr = adata.obs[config["cluster_column"]].astype(str).values
        max_len = 20
        cell_ids_padded = np.array([s[:max_len].ljust(max_len) for s in cell_ids_arr], dtype=f"<U{max_len}")
        clusters_padded = np.array([s[:max_len].ljust(max_len) for s in clusters_arr], dtype=f"<U{max_len}")
        print(f"\n[3/7] Creating shared memory...")
        shm_data = shared_memory.SharedMemory(create=True, size=data_arr.nbytes)
        shm_indices = shared_memory.SharedMemory(create=True, size=indices_arr.nbytes)
        shm_indptr = shared_memory.SharedMemory(create=True, size=indptr_arr.nbytes)
        shm_cell_ids = shared_memory.SharedMemory(create=True, size=cell_ids_padded.nbytes)
        shm_clusters = shared_memory.SharedMemory(create=True, size=clusters_padded.nbytes)
        np.ndarray(data_arr.shape, dtype=data_arr.dtype, buffer=shm_data.buf)[:] = data_arr
        np.ndarray(indices_arr.shape, dtype=indices_arr.dtype, buffer=shm_indices.buf)[:] = indices_arr
        np.ndarray(indptr_arr.shape, dtype=indptr_arr.dtype, buffer=shm_indptr.buf)[:] = indptr_arr
        np.ndarray(cell_ids_padded.shape, dtype=cell_ids_padded.dtype, buffer=shm_cell_ids.buf)[:] = cell_ids_padded
        np.ndarray(clusters_padded.shape, dtype=clusters_padded.dtype, buffer=shm_clusters.buf)[:] = clusters_padded
        shm_handles = [shm_data, shm_indices, shm_indptr, shm_cell_ids, shm_clusters]
        init_args = (
            shm_data.name, shm_indices.name, shm_indptr.name,
            shm_cell_ids.name, shm_clusters.name,
            data_arr.shape, data_arr.dtype, indices_arr.shape, indices_arr.dtype,
            indptr_arr.shape, indptr_arr.dtype, cell_ids_padded.shape, clusters_padded.shape,
            X_csr.shape, list(adata.var_names), config["cluster_column"], config["cell_id_column"],
            config["markers"], config["top_genes"],
            args.skip_mast,
            args.skip_gsea,
            args.order_only,
        )
        initializer = init_worker
    else:
        print(f"\n[2/7] Workers will load adata from {config['adata_path']}...")
        print(f"\n[3/7] Skipping shared memory (not available).")
        init_args = (
            config["adata_path"],
            config["cluster_column"],
            config["cell_id_column"],
            config["markers"],
            config["top_genes"],
            args.skip_mast,
            args.skip_gsea,
            args.order_only,
        )
        initializer = init_worker_load_from_path

    print(f"\n[4/7] Running {len(tasks)} tasks with {args.n_workers} workers...")
    print("      (Bottlenecks: MAST and LN per cluster; expect ~1–5+ min per task depending on n_genes/n_clusters.)")
    start = time.time()
    all_records = []
    try:
        ctx = multiprocessing.get_context("spawn")
        with ctx.Pool(processes=args.n_workers, initializer=initializer, initargs=init_args) as pool:
            for i, result in enumerate(pool.imap_unordered(process_task, tasks)):
                records, task_time = result
                all_records.extend(records)
                elapsed = time.time() - start
                rate = (i + 1) / elapsed if elapsed > 0 else 0
                eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                print(f"  Task {i+1}/{len(tasks)} done in {task_time:.0f}s  |  elapsed: {elapsed:.0f}s  ETA: {eta:.0f}s")
    finally:
        for shm in shm_handles:
            try:
                shm.close()
                shm.unlink()
            except Exception:
                pass

    total_time = time.time() - start
    print(f"\n  Done in {total_time:.1f}s ({len(tasks)/total_time:.1f} tasks/s)")

    df_results = pd.DataFrame(all_records)
    print(f"\n[5/7] Saving results to {args.output_dir}/...")
    df_results.to_csv(os.path.join(args.output_dir, "subsampling_results_raw.csv"), index=False)
    agg_cols = {
        "jaccard_top": ["mean", "std"],
        "jaccard_all": ["mean", "std"],
        "avg_abs_lfc": ["mean", "std"],
        "n_sig_genes": ["mean", "std"],
        "n_genes_de_table": ["mean", "std"],
        "frac_de_genes_valid_score": ["mean", "std"],
        "de_time_s": ["mean", "std"],
    }
    if "gsea_nes" in df_results.columns:
        agg_cols["gsea_nes"] = ["mean", "std"]
    df_agg = df_results.groupby(["fraction", "method", "cluster"]).agg(agg_cols).reset_index()
    df_agg.columns = ["_".join(c).strip("_") for c in df_agg.columns]
    df_agg.to_csv(os.path.join(args.output_dir, "subsampling_results_aggregated.csv"), index=False)

    print(f"\n[6/7] Generating plots (per-cluster and combined)...")
    # Metrics to plot; each gets a combined plot and a per-cluster faceted plot
    plot_metrics = [
        ("jaccard_top", "jaccard_top_vs_subsampling.png"),
        ("jaccard_all", "jaccard_all_vs_subsampling.png"),
        ("auprc", "auprc_vs_subsampling.png"),
    ]
    if "gsea_nes" in df_results.columns:
        plot_metrics.append(("gsea_nes", "gsea_nes_vs_subsampling.png"))
    for y_col, fname in plot_metrics:
        if y_col not in df_results.columns:
            continue
        # Combined (all clusters) - use pandas/matplotlib to avoid seaborn boxplot bugs
        fig, ax = plt.subplots(figsize=(12, 6))
        _plot_box_pandas(df_results, "fraction", y_col, "method", ax=ax)
        ax.set_xlabel("Sub-sampling fraction (p)")
        ax.set_ylabel(y_col)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, fname), dpi=300)
        plt.close()
        # Per-cluster: one panel per cluster
        clusters = sorted(df_results["cluster"].unique())
        fig, axes = plt.subplots(1, len(clusters), figsize=(6 * len(clusters), 5), sharey=True)
        if len(clusters) == 1:
            axes = [axes]
        for ax, cl in zip(axes, clusters):
            sub = df_results[df_results["cluster"] == cl]
            _plot_box_pandas(sub, "fraction", y_col, "method", ax=ax)
            ax.set_title(str(cl))
            ax.set_xlabel("Sub-sampling fraction (p)")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel(y_col)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, fname.replace(".png", "_by_cluster.png")), dpi=300)
        plt.close()
    # Precision@k multi-panel (combined)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for ax, k in zip(axes.flat, [10, 20, 50, 100]):
        col = f"precision_at_{k}"
        if col in df_results.columns:
            _plot_box_pandas(df_results, "fraction", col, "method", ax=ax)
            ax.set_ylabel(f"Precision@{k}")
        ax.set_xlabel("Sub-sampling fraction (p)")
        ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "precision_at_k_vs_subsampling.png"), dpi=300)
    plt.close()
    # Precision@k by cluster
    for k in [10, 20, 50, 100]:
        col = f"precision_at_{k}"
        if col not in df_results.columns:
            continue
        clusters = sorted(df_results["cluster"].unique())
        fig, axes = plt.subplots(1, len(clusters), figsize=(6 * len(clusters), 4), sharey=True)
        if len(clusters) == 1:
            axes = [axes]
        for ax, cl in zip(axes, clusters):
            sub = df_results[df_results["cluster"] == cl]
            _plot_box_pandas(sub, "fraction", col, "method", ax=ax)
            ax.set_title(str(cl))
            ax.set_xlabel("Sub-sampling fraction (p)")
            ax.set_ylabel(f"Precision@{k}")
            ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"precision_at_{k}_vs_subsampling_by_cluster.png"), dpi=300)
        plt.close()
    # AUPRC vs Jaccard scatter
    plt.figure(figsize=(10, 8))
    for method in df_results["method"].unique():
        subset = df_results[df_results["method"] == method]
        plt.scatter(subset["jaccard_top"], subset["auprc"], alpha=0.3, label=method, s=20)
    plt.xlabel(f"Jaccard (top {config['top_genes']} genes)")
    plt.ylabel("AUPRC")
    plt.legend(title="Method")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "auprc_vs_jaccard_scatter.png"), dpi=300)
    plt.close()
    df_lfc = df_results.drop_duplicates(subset=["fraction", "replicate", "method", "cluster"])
    for y_col, fname in [
        ("avg_abs_lfc", "avg_lfc_vs_subsampling.png"),
        ("n_sig_genes", "n_sig_genes_vs_subsampling.png"),
        ("n_genes_de_table", "n_genes_de_table_vs_subsampling.png"),
        ("frac_de_genes_valid_score", "frac_de_genes_valid_score_vs_subsampling.png"),
    ]:
        if y_col not in df_lfc.columns:
            continue
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_box_pandas(df_lfc, "fraction", y_col, "method", ax=ax)
        ax.set_xlabel("Sub-sampling fraction (p)")
        ax.set_ylabel(y_col)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, fname), dpi=300)
        plt.close()
        # Per-cluster
        clusters = sorted(df_lfc["cluster"].unique())
        fig, axes = plt.subplots(1, len(clusters), figsize=(6 * len(clusters), 5), sharey=True)
        if len(clusters) == 1:
            axes = [axes]
        for ax, cl in zip(axes, clusters):
            sub = df_lfc[df_lfc["cluster"] == cl]
            _plot_box_pandas(sub, "fraction", y_col, "method", ax=ax)
            ax.set_title(str(cl))
            ax.set_xlabel("Sub-sampling fraction (p)")
            ax.grid(True, axis="y", alpha=0.3)
        axes[0].set_ylabel(y_col)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, fname.replace(".png", "_by_cluster.png")), dpi=300)
        plt.close()
    # DE time per method (one value per task per method)
    if "de_time_s" in df_results.columns:
        df_time = df_results.drop_duplicates(subset=["fraction", "replicate", "method"])
        fig, ax = plt.subplots(figsize=(10, 6))
        _plot_box_pandas(df_time, "fraction", "de_time_s", "method", ax=ax)
        ax.set_xlabel("Sub-sampling fraction (p)")
        ax.set_ylabel("DE time (s)")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "de_time_s_vs_subsampling.png"), dpi=300)
        plt.close()

    print(f"\n[7/7] Summary")
    # Per-cluster: two sets of measures (one per cluster)
    for cluster_id in df_results["cluster"].unique():
        sub = df_results[df_results["cluster"] == cluster_id]
        print(f"\n--- Cluster: {cluster_id} ---")
        print(sub.groupby(["fraction", "method"])["jaccard_top"].mean().unstack().to_string())
    if "gsea_nes" in df_results.columns:
        print("\nGSEA NES (mean) by cluster and method:")
        print(df_results.groupby(["cluster", "fraction", "method"])["gsea_nes"].mean().unstack().to_string())
    if "de_time_s" in df_results.columns:
        df_time = df_results.drop_duplicates(subset=["fraction", "replicate", "method"])
        print("\nMean DE time (s) by method:")
        print(df_time.groupby(["fraction", "method"])["de_time_s"].mean().unstack().to_string())
    print(f"\nDone. Results in {args.output_dir}/")


if __name__ == "__main__":
    main()
