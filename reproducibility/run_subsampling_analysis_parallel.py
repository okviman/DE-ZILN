#!/usr/bin/env python3
"""
Parallelized sub-sampling effect analysis using shared memory and nested sampling.

Uses a nested Binomial sampling design: the gene set is fixed at the lowest
valid fraction (p_base), and spot selections are extended to higher fractions
using conditional Bernoulli draws. This ensures the gene universe is identical
across all p values for a given replicate, making metrics directly comparable.

Uses de_utils for DE (LN, t-test, wilcoxon, MAST) and evaluation_utils for
all metrics (Jaccard, AUPRC, precision@k, avg |LFC|, n_sig_genes).
Input adata.X = raw counts (spot-level); after subsampling we aggregate to
cell-level counts and run the same analysis as the celltype pipeline.

Usage:
    python run_subsampling_analysis_parallel.py --config config.yaml [--output-dir output/] [--n-workers 8]

Optional in config (defaults used if omitted):
  - min_cells_per_gene (int): genes must be expressed in at least that many cells in
    *each* group to be kept; otherwise dropped before DE.
  - min_spots_per_cell (int, default 5): cells with fewer than this many 2um spots after
    subsampling are excluded.
  - min_counts_after_subsampling (int, default 20): cells whose chosen 2um spots have total
    UMI count less than this are excluded (not included in the aggregated matrix).
  - min_genes (int, default 500): minimum number of genes after filtering for a base
    sampling to be considered valid.
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
import json
import pickle
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
# Defaults for optional config keys (overridden by YAML if present)
DEFAULT_MIN_SPOTS_PER_CELL = 5
DEFAULT_MIN_COUNTS_AFTER_SUBSAMPLING = 20
DEFAULT_MIN_GENES = 500
DEFAULT_MAX_CANDIDATES_FACTOR = 10


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


def find_valid_base_samplings(X_csr, cell_ids, clusters, var_names, fractions_sorted,
                               n_replicates, min_genes, min_spots_per_cell,
                               min_counts_after_subsampling, min_cells_per_gene,
                               max_candidates_factor=DEFAULT_MAX_CANDIDATES_FACTOR):
    """Find n_replicates valid base samplings at the lowest possible fraction.

    Uses Binomial (independent Bernoulli per spot) sampling.  For each candidate,
    stores the exact spot indices selected for every cell (including cells that
    fail filters, so they can re-enter at higher fractions) and the gene set
    that survives iterative filtering.

    Returns
    -------
    p_base : float
    valid_samplings : list of dicts with keys 'spots', 'genes', 'n_genes', 'n_cells',
        'valid_cells'
    """
    unique_cells = np.unique(cell_ids)
    cell_spot_map = {}
    cell_to_cluster = {}
    for cid in unique_cells:
        indices = np.where(cell_ids == cid)[0]
        cell_spot_map[cid] = indices
        cell_to_cluster[cid] = clusters[indices[0]]

    for p in fractions_sorted:
        valid_samplings = []
        max_candidates = max_candidates_factor * n_replicates
        print(f"  Trying p={p}: up to {max_candidates} candidates...")

        for seed in range(max_candidates):
            rng = np.random.RandomState(seed)

            spots_per_cell = {}
            for cid in unique_cells:
                spot_indices = cell_spot_map[cid]
                mask = rng.random(len(spot_indices)) < p
                spots_per_cell[cid] = spot_indices[mask]

            count_list = []
            cell_order = []
            cluster_assignments = []
            for cid in unique_cells:
                sel = spots_per_cell[cid]
                if len(sel) < min_spots_per_cell:
                    continue
                cell_counts = np.asarray(X_csr[sel, :].sum(axis=0)).flatten()
                if cell_counts.sum() < min_counts_after_subsampling:
                    continue
                count_list.append(cell_counts)
                cell_order.append(cid)
                cluster_assignments.append(cell_to_cluster[cid])

            if len(count_list) == 0:
                continue

            aggregated = np.array(count_list, dtype=np.float32)
            adata_tmp = sc.AnnData(
                X=sparse.csr_matrix(aggregated),
                obs=pd.DataFrame(
                    {"_cluster": cluster_assignments},
                    index=[str(c) for c in cell_order],
                ),
                var=pd.DataFrame(index=var_names),
            )
            adata_tmp.obs["_cluster"] = pd.Categorical(adata_tmp.obs["_cluster"])

            for _ in range(50):
                n_obs_before, n_vars_before = adata_tmp.n_obs, adata_tmp.n_vars
                Xf = adata_tmp.X
                cl_vals = adata_tmp.obs["_cluster"].values

                if min_cells_per_gene is not None and min_cells_per_gene > 0:
                    keep_genes = np.ones(adata_tmp.n_vars, dtype=bool)
                    for cl in np.unique(cl_vals):
                        mask_c = cl_vals == cl
                        if sparse.issparse(Xf):
                            ncpg = np.asarray((Xf[mask_c, :] > 0).sum(axis=0)).ravel()
                        else:
                            ncpg = (Xf[mask_c, :] > 0).sum(axis=0)
                        keep_genes &= (ncpg >= min_cells_per_gene)
                    adata_tmp = adata_tmp[:, keep_genes].copy()
                    Xf = adata_tmp.X

                if min_counts_after_subsampling is not None and min_counts_after_subsampling > 0:
                    if sparse.issparse(Xf):
                        total_umi = np.asarray(Xf.sum(axis=1)).ravel()
                    else:
                        total_umi = Xf.sum(axis=1)
                    keep_cells = total_umi >= min_counts_after_subsampling
                    adata_tmp = adata_tmp[keep_cells, :].copy()

                if adata_tmp.n_obs == 0 or adata_tmp.n_vars == 0:
                    break
                if adata_tmp.n_obs == n_obs_before and adata_tmp.n_vars == n_vars_before:
                    break

            if adata_tmp.n_vars < min_genes or adata_tmp.n_obs == 0:
                continue

            clusters_present = adata_tmp.obs["_cluster"].value_counts()
            if (clusters_present >= 1).sum() < 2:
                continue

            valid_cells = [idx.strip() for idx in adata_tmp.obs.index]
            valid_samplings.append({
                'spots': spots_per_cell,
                'genes': list(adata_tmp.var_names),
                'n_genes': adata_tmp.n_vars,
                'n_cells': adata_tmp.n_obs,
                'valid_cells': valid_cells,
            })
            print(f"    Seed {seed}: VALID ({adata_tmp.n_vars} genes, "
                  f"{adata_tmp.n_obs} cells) [{len(valid_samplings)}/{n_replicates}]")

            if len(valid_samplings) == n_replicates:
                break

        if len(valid_samplings) == n_replicates:
            print(f"  => p_base = {p} ({n_replicates} valid samplings found)")
            return p, valid_samplings
        else:
            print(f"  WARNING: p={p} only produced "
                  f"{len(valid_samplings)}/{n_replicates} valid samplings, skipping")

    raise RuntimeError(
        f"No fraction produced {n_replicates} valid samplings with >= {min_genes} genes. "
        f"Tried fractions: {fractions_sorted}"
    )


def extend_spots_binomial(base_spots, cell_spot_map, p_base, p, rng,
                          restrict_to_base_cells=False):
    """Extend base spot selections to a higher fraction using conditional Bernoulli.

    Each spot NOT in the base set is independently included with probability
    q = (p - p_base) / (1 - p_base), giving marginal inclusion probability p.

    If restrict_to_base_cells is True, cells that had zero spots at p_base are
    kept empty (no new spots drawn), so only cells already present at p_base
    can appear at higher fractions.
    """
    q = (p - p_base) / (1.0 - p_base)
    extended = {}
    for cell_id, all_spots in cell_spot_map.items():
        base = base_spots.get(cell_id, np.array([], dtype=np.intp))
        if len(base) == 0:
            if restrict_to_base_cells:
                extended[cell_id] = np.array([], dtype=np.intp)
            else:
                mask = rng.random(len(all_spots)) < q
                extended[cell_id] = all_spots[mask]
        elif len(base) >= len(all_spots):
            extended[cell_id] = all_spots.copy()
        else:
            base_set = set(base.tolist())
            remaining_mask = np.array([s not in base_set for s in all_spots])
            remaining = all_spots[remaining_mask]
            if len(remaining) == 0:
                extended[cell_id] = base.copy()
            else:
                add_mask = rng.random(len(remaining)) < q
                extended[cell_id] = np.concatenate([base, remaining[add_mask]])
    return extended


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
    min_cells_per_gene=None,
    min_spots_per_cell=None,
    min_counts_after_subsampling=None,
    p_base=None,
    base_samplings_path=None,
    same_gene_sets_all_p=False,
    same_cells_all_p=False,
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
    _shared_data["min_cells_per_gene"] = min_cells_per_gene
    _shared_data["min_spots_per_cell"] = min_spots_per_cell
    _shared_data["min_counts_after_subsampling"] = min_counts_after_subsampling
    _shared_data["p_base"] = p_base
    _shared_data["same_gene_sets_all_p"] = same_gene_sets_all_p
    _shared_data["same_cells_all_p"] = same_cells_all_p
    if base_samplings_path is not None:
        with open(base_samplings_path, "rb") as f:
            _shared_data["base_samplings"] = pickle.load(f)


def init_worker_load_from_path(adata_path, cluster_column, cell_id_column, markers, top_genes, skip_mast=False, skip_gsea=False, order_only=False, min_cells_per_gene=None, min_spots_per_cell=None, min_counts_after_subsampling=None, p_base=None, base_samplings_path=None, same_gene_sets_all_p=False, same_cells_all_p=False):
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
    _shared_data["min_cells_per_gene"] = min_cells_per_gene
    _shared_data["min_spots_per_cell"] = min_spots_per_cell
    _shared_data["min_counts_after_subsampling"] = min_counts_after_subsampling
    _shared_data["p_base"] = p_base
    _shared_data["same_gene_sets_all_p"] = same_gene_sets_all_p
    _shared_data["same_cells_all_p"] = same_cells_all_p
    if base_samplings_path is not None:
        with open(base_samplings_path, "rb") as f:
            _shared_data["base_samplings"] = pickle.load(f)


def process_task(task):
    """Process a single (fraction, replicate) task with nested sampling.

    Uses the base spot selections from p_base (loaded via base_samplings) and extends
    them to the target fraction using conditional Bernoulli draws.  When
    same_gene_sets_all_p is True (or at p_base), the gene set is fixed per replicate.
    Otherwise, genes are re-discovered via iterative min_cells_per_gene filtering.
    """
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
    min_spots_per_cell = _shared_data.get("min_spots_per_cell", DEFAULT_MIN_SPOTS_PER_CELL)
    min_counts_after_subsampling = _shared_data.get("min_counts_after_subsampling", DEFAULT_MIN_COUNTS_AFTER_SUBSAMPLING)
    p_base = _shared_data["p_base"]
    base_samplings = _shared_data["base_samplings"]

    # Lazy-init cached lookups (computed once per worker, reused across tasks)
    if "cell_spot_map" not in _shared_data:
        unique_cells = np.unique(cell_ids)
        csm = {}
        ctc = {}
        for c in unique_cells:
            c_stripped = c.strip()
            csm[c_stripped] = np.where(cell_ids == c)[0]
            ctc[c_stripped] = clusters[np.where(cell_ids == c)[0][0]].strip()
        _shared_data["cell_spot_map"] = csm
        _shared_data["cell_to_cluster"] = ctc
    if "var_name_to_idx" not in _shared_data:
        _shared_data["var_name_to_idx"] = {g: i for i, g in enumerate(var_names)}

    cell_spot_map = _shared_data["cell_spot_map"]
    cell_to_cluster = _shared_data["cell_to_cluster"]
    var_name_to_idx = _shared_data["var_name_to_idx"]

    base_sampling = base_samplings[replicate]
    base_spots_dict = base_sampling['spots']
    min_cells_per_gene = _shared_data.get("min_cells_per_gene")
    same_gene_sets_all_p = _shared_data.get("same_gene_sets_all_p", False)
    same_cells_all_p = _shared_data.get("same_cells_all_p", False)

    rng = np.random.RandomState(replicate * 100000 + int(round(fraction * 100000)))

    at_base = abs(fraction - p_base) < 1e-9
    if at_base:
        selected_spots = base_spots_dict
    else:
        selected_spots = extend_spots_binomial(
            base_spots_dict, cell_spot_map, p_base, fraction, rng,
            restrict_to_base_cells=same_cells_all_p,
        )

    use_fixed_genes = same_gene_sets_all_p or at_base

    if same_cells_all_p and not at_base:
        cells_to_iterate = base_sampling['valid_cells']
    else:
        cells_to_iterate = list(cell_spot_map.keys())

    if use_fixed_genes:
        gene_set_list = base_sampling['genes']
        gene_indices = np.array([var_name_to_idx[g] for g in gene_set_list])

        count_list = []
        cell_order = []
        cluster_assignments = []
        for cell_id in cells_to_iterate:
            sel = selected_spots.get(cell_id, np.array([], dtype=np.intp))
            if len(sel) < min_spots_per_cell:
                continue
            cell_counts = np.asarray(X[sel, :][:, gene_indices].sum(axis=0)).flatten()
            if cell_counts.sum() < min_counts_after_subsampling:
                continue
            count_list.append(cell_counts)
            cell_order.append(cell_id)
            cluster_assignments.append(cell_to_cluster[cell_id])

        if len(count_list) == 0:
            return [], time.time() - t0

        aggregated_counts = np.array(count_list, dtype=np.float32)
        adata_agg = sc.AnnData(
            X=sparse.csr_matrix(aggregated_counts),
            obs=pd.DataFrame(
                {cell_id_column: cell_order, cluster_column: cluster_assignments},
                index=[str(c) for c in cell_order],
            ),
            var=pd.DataFrame(index=gene_set_list),
        )
    else:
        count_list = []
        cell_order = []
        cluster_assignments = []
        for cell_id in cells_to_iterate:
            sel = selected_spots.get(cell_id, np.array([], dtype=np.intp))
            if len(sel) < min_spots_per_cell:
                continue
            cell_counts = np.asarray(X[sel, :].sum(axis=0)).flatten()
            count_list.append(cell_counts)
            cell_order.append(cell_id)
            cluster_assignments.append(cell_to_cluster[cell_id])

        if len(count_list) == 0:
            return [], time.time() - t0

        aggregated_counts = np.array(count_list, dtype=np.float32)
        adata_agg = sc.AnnData(
            X=sparse.csr_matrix(aggregated_counts),
            obs=pd.DataFrame(
                {cell_id_column: cell_order, cluster_column: cluster_assignments},
                index=[str(c) for c in cell_order],
            ),
            var=pd.DataFrame(index=var_names),
        )
        adata_agg.obs[cluster_column] = pd.Categorical(adata_agg.obs[cluster_column])

        for _ in range(50):
            n_obs_before, n_vars_before = adata_agg.n_obs, adata_agg.n_vars
            Xf = adata_agg.X
            if min_cells_per_gene is not None and min_cells_per_gene > 0:
                keep_genes = np.ones(adata_agg.n_vars, dtype=bool)
                cl_vals = adata_agg.obs[cluster_column].values
                for cl in np.unique(cl_vals):
                    mask_c = cl_vals == cl
                    if sparse.issparse(Xf):
                        ncpg = np.asarray((Xf[mask_c, :] > 0).sum(axis=0)).ravel()
                    else:
                        ncpg = (Xf[mask_c, :] > 0).sum(axis=0)
                    keep_genes &= (ncpg >= min_cells_per_gene)
                adata_agg = adata_agg[:, keep_genes].copy()
                Xf = adata_agg.X
            if min_counts_after_subsampling is not None and min_counts_after_subsampling > 0:
                if sparse.issparse(Xf):
                    total_umi = np.asarray(Xf.sum(axis=1)).ravel()
                else:
                    total_umi = Xf.sum(axis=1)
                keep_cells = total_umi >= min_counts_after_subsampling
                adata_agg = adata_agg[keep_cells, :].copy()
            if adata_agg.n_obs == 0 or adata_agg.n_vars == 0:
                break
            if adata_agg.n_obs == n_obs_before and adata_agg.n_vars == n_vars_before:
                break

        gene_set_list = list(adata_agg.var_names)

    n_genes_used = len(gene_set_list)
    adata_agg.obs[cluster_column] = pd.Categorical(adata_agg.obs[cluster_column])

    if adata_agg.n_vars == 0 or adata_agg.n_obs == 0:
        return [], time.time() - t0

    clusters_present = adata_agg.obs[cluster_column].value_counts()
    if (clusters_present >= 1).sum() < 2:
        return [], time.time() - t0

    cluster_cell_counts = clusters_present.to_dict()
    n_cells_total = adata_agg.n_obs

    # DE via de_utils
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t_layers = time.time()
        prepare_adata_layers(adata_agg)
        time_layers_s = time.time() - t_layers
        t_ln = time.time()
        run_ln_de(adata_agg, cluster_column, key_added="ln_test", layer="counts", sparse=True)
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

    cluster_list = sorted(adata_agg.obs[cluster_column].unique())
    records = []

    for cluster in cluster_list:
        cluster_key = str(cluster).strip()
        marker_set = set(markers.get(cluster_key, []))
        if not marker_set:
            continue
        n_cells_in_cluster = cluster_cell_counts.get(cluster, 0)
        n_cells_in_other = n_cells_total - n_cells_in_cluster
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
            "n_genes_used": n_genes_used, "p_base": p_base,
            "n_cells_cluster": n_cells_in_cluster, "n_cells_other": n_cells_in_other,
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
            "n_genes_used": n_genes_used, "p_base": p_base,
            "n_cells_cluster": n_cells_in_cluster, "n_cells_other": n_cells_in_other,
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
            "n_genes_used": n_genes_used, "p_base": p_base,
            "n_cells_cluster": n_cells_in_cluster, "n_cells_other": n_cells_in_other,
        })
        # MAST
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
                "cluster": cluster, "method": "MAST",
                "jaccard_top": jt, "jaccard_all": ja, "auprc": au,
                "precision_at_10": pk.get(10, np.nan), "precision_at_20": pk.get(20, np.nan),
                "precision_at_50": pk.get(50, np.nan), "precision_at_100": pk.get(100, np.nan),
                "avg_abs_lfc": avg_lfc_mast, "n_sig_genes": n_sig_mast, "gsea_nes": gsea_mast,
                "n_genes_de_table": n_genes_mast, "frac_de_genes_valid_score": frac_valid_mast,
                "de_time_s": time_mast_s,
                "fraction": fraction, "replicate": replicate,
                "n_genes_used": n_genes_used, "p_base": p_base,
                "n_cells_cluster": n_cells_in_cluster, "n_cells_other": n_cells_in_other,
            })

    task_time = time.time() - t0
    return records, task_time


def main():
    parser = argparse.ArgumentParser(description="Parallelized sub-sampling analysis with nested Binomial sampling")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="output_subsampling_parallel", help="Output directory")
    parser.add_argument("--n-workers", type=int, default=8, help="Number of parallel workers")
    parser.add_argument("--skip-mast", action="store_true", help="Skip MAST DE (faster runs for debugging)")
    parser.add_argument("--skip-gsea", action="store_true", help="Skip GSEA NES (avoids gseapy errors/noise)")
    parser.add_argument("--order-only", action="store_true", help="Use only rank order for AUPR/GSEA (replace scores with ranks)")
    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("PARALLELIZED SUB-SAMPLING ANALYSIS -- NESTED BINOMIAL DESIGN")
    print("=" * 80)
    print(f"Using {args.n_workers} workers, DE and metrics from de_utils / evaluation_utils")

    config = load_config(args.config)
    os.makedirs(args.output_dir, exist_ok=True)
    min_spots_per_cell = config.get("min_spots_per_cell", DEFAULT_MIN_SPOTS_PER_CELL)
    min_counts_after_subsampling = config.get("min_counts_after_subsampling", DEFAULT_MIN_COUNTS_AFTER_SUBSAMPLING)
    min_cells_per_gene = config.get("min_cells_per_gene")
    min_genes = config.get("min_genes", DEFAULT_MIN_GENES)
    same_gene_sets_all_p = config.get("same_gene_sets_all_p", False)
    same_cells_all_p = config.get("same_cells_all_p", False)
    n_replicates = config["n_replicates"]
    fractions = sorted(config["subsampling_fractions"])

    print(f"\n[1/8] Config loaded. Fractions: {fractions}, Replicates: {n_replicates}")
    print(f"  min_spots_per_cell: {min_spots_per_cell}")
    print(f"  min_counts_after_subsampling: {min_counts_after_subsampling}")
    print(f"  min_genes: {min_genes} (minimum genes after filtering for a valid base sampling)")
    print(f"  same_gene_sets_all_p: {same_gene_sets_all_p}"
          f" ({'gene sets fixed from p_base' if same_gene_sets_all_p else 'genes re-discovered per p'})")
    print(f"  same_cells_all_p: {same_cells_all_p}"
          f" ({'cells restricted to p_base set' if same_cells_all_p else 'new cells can appear at higher p'})")
    if args.skip_mast:
        print("  --skip-mast: MAST DE disabled (faster run)")
    if args.order_only:
        print("  --order-only: AUPR/GSEA use rank order only (ignore score magnitude)")
    if min_cells_per_gene is not None and min_cells_per_gene > 0:
        print(f"  min_cells_per_gene: {min_cells_per_gene}")
    if not USE_SHARED_MEMORY:
        print("  (Python < 3.8: each worker will load adata from disk; no shared memory)")

    # ---- Load adata (needed for pre-sampling AND shared memory) ----
    print(f"\n[2/8] Loading adata from {config['adata_path']}...")
    adata = sc.read_h5ad(config["adata_path"])
    if sparse.issparse(adata.X):
        X_csr = adata.X.tocsr()
    else:
        X_csr = sparse.csr_matrix(adata.X)
    cell_ids_arr = adata.obs[config["cell_id_column"]].astype(str).values
    clusters_arr = adata.obs[config["cluster_column"]].astype(str).values
    var_names_list = list(adata.var_names)
    print(f"  {X_csr.shape[0]} spots x {X_csr.shape[1]} genes, "
          f"{len(np.unique(cell_ids_arr))} cells")

    # ---- Pre-sampling phase ----
    print(f"\n[3/8] Pre-sampling phase: finding {n_replicates} valid base samplings "
          f"(min_genes={min_genes})...")
    t_presample = time.time()
    p_base, base_samplings = find_valid_base_samplings(
        X_csr, cell_ids_arr, clusters_arr, var_names_list, fractions,
        n_replicates, min_genes, min_spots_per_cell,
        min_counts_after_subsampling, min_cells_per_gene,
    )
    t_presample = time.time() - t_presample

    valid_fractions = [p for p in fractions if p >= p_base]
    skipped_fractions = [p for p in fractions if p < p_base]
    if skipped_fractions:
        print(f"  Skipped fractions (< p_base): {skipped_fractions}")
    gene_set_sizes = [s['n_genes'] for s in base_samplings]
    print(f"  Gene set sizes across replicates: min={min(gene_set_sizes)}, "
          f"max={max(gene_set_sizes)}, mean={np.mean(gene_set_sizes):.0f}")
    print(f"  Pre-sampling took {t_presample:.1f}s")

    base_samplings_path = os.path.join(args.output_dir, "_base_samplings.pkl")
    with open(base_samplings_path, "wb") as f:
        pickle.dump(base_samplings, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ---- Build tasks ----
    tasks = []
    for fraction in valid_fractions:
        for rep in range(n_replicates):
            tasks.append((fraction, rep))

    # ---- Shared memory / worker init ----
    shm_handles = []
    if USE_SHARED_MEMORY:
        print(f"\n[4/8] Creating shared memory...")
        data_arr = np.array(X_csr.data, dtype=np.float32)
        indices_arr = np.array(X_csr.indices, dtype=np.int32)
        indptr_arr = np.array(X_csr.indptr, dtype=np.int64)
        max_len = 20
        cell_ids_padded = np.array([s[:max_len].ljust(max_len) for s in cell_ids_arr], dtype=f"<U{max_len}")
        clusters_padded = np.array([s[:max_len].ljust(max_len) for s in clusters_arr], dtype=f"<U{max_len}")
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
            X_csr.shape, var_names_list, config["cluster_column"], config["cell_id_column"],
            config["markers"], config["top_genes"],
            args.skip_mast, args.skip_gsea, args.order_only,
            min_cells_per_gene,
            min_spots_per_cell,
            min_counts_after_subsampling,
            p_base,
            base_samplings_path,
            same_gene_sets_all_p,
            same_cells_all_p,
        )
        initializer = init_worker
    else:
        print(f"\n[4/8] Workers will load adata from disk (no shared memory).")
        init_args = (
            config["adata_path"],
            config["cluster_column"], config["cell_id_column"],
            config["markers"], config["top_genes"],
            args.skip_mast, args.skip_gsea, args.order_only,
            min_cells_per_gene,
            min_spots_per_cell,
            min_counts_after_subsampling,
            p_base,
            base_samplings_path,
            same_gene_sets_all_p,
            same_cells_all_p,
        )
        initializer = init_worker_load_from_path

    del adata, X_csr

    # ---- Parallel DE ----
    print(f"\n[5/8] Running {len(tasks)} tasks ({len(valid_fractions)} fractions x "
          f"{n_replicates} replicates) with {args.n_workers} workers...")
    if same_gene_sets_all_p:
        print("      (Gene sets are fixed per replicate from p_base; spots are nested.)")
    else:
        print("      (Genes re-discovered per p via min_cells_per_gene filtering; spots are nested.)")
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
        try:
            os.remove(base_samplings_path)
        except OSError:
            pass

    total_time = time.time() - start
    print(f"\n  Done in {total_time:.1f}s ({len(tasks)/total_time:.1f} tasks/s)")

    # ---- Save results ----
    df_results = pd.DataFrame(all_records)
    print(f"\n[6/8] Saving results to {args.output_dir}/...")
    df_results.to_csv(os.path.join(args.output_dir, "subsampling_results_raw.csv"), index=False)
    agg_cols = {
        "jaccard_top": ["mean", "std"],
        "jaccard_all": ["mean", "std"],
        "avg_abs_lfc": ["mean", "std"],
        "n_sig_genes": ["mean", "std"],
        "n_genes_de_table": ["mean", "std"],
        "n_genes_used": ["mean", "std"],
        "n_cells_cluster": ["mean", "std"],
        "n_cells_other": ["mean", "std"],
        "frac_de_genes_valid_score": ["mean", "std"],
        "de_time_s": ["mean", "std"],
    }
    if "gsea_nes" in df_results.columns:
        agg_cols["gsea_nes"] = ["mean", "std"]
    df_agg = df_results.groupby(["fraction", "method", "cluster"]).agg(agg_cols).reset_index()
    df_agg.columns = ["_".join(c).strip("_") for c in df_agg.columns]
    df_agg.to_csv(os.path.join(args.output_dir, "subsampling_results_aggregated.csv"), index=False)

    metadata = {
        "p_base": p_base,
        "valid_fractions": valid_fractions,
        "skipped_fractions": skipped_fractions,
        "n_replicates": n_replicates,
        "min_genes": min_genes,
        "min_spots_per_cell": min_spots_per_cell,
        "min_counts_after_subsampling": min_counts_after_subsampling,
        "min_cells_per_gene": min_cells_per_gene,
        "same_gene_sets_all_p": same_gene_sets_all_p,
        "same_cells_all_p": same_cells_all_p,
        "pre_sampling_time_s": round(t_presample, 1),
        "per_replicate_gene_set_sizes": gene_set_sizes,
    }
    with open(os.path.join(args.output_dir, "subsampling_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # ---- Plots ----
    print(f"\n[7/8] Generating plots (per-cluster and combined)...")
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
        ("n_genes_used", "n_genes_used_vs_subsampling.png"),
        ("frac_de_genes_valid_score", "frac_de_genes_valid_score_vs_subsampling.png"),
        ("n_cells_cluster", "n_cells_cluster_vs_subsampling.png"),
        ("n_cells_other", "n_cells_other_vs_subsampling.png"),
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

    # ---- Summary ----
    print(f"\n[8/8] Summary")
    print(f"  p_base: {p_base}")
    print(f"  Valid fractions: {valid_fractions}")
    if skipped_fractions:
        print(f"  Skipped fractions: {skipped_fractions}")
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
