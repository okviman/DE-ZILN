"""
Shared utilities for differential expression in the reproducibility folder.

Provides: layer preparation (counts, norm_counts, log1p_norm), LN test, t-test,
wilcoxon, and MAST. Used by celltype, clustering, and subsampling scripts.
"""

import os
import importlib.util
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

# Optional R/MAST
try:
    import rpy2.robjects as ro
    from rpy2.robjects import pandas2ri
    from rpy2.robjects.packages import importr
    from rpy2.robjects.conversion import localconverter
    _HAS_RPY2 = True
except ImportError:
    _HAS_RPY2 = False

_scanpy_wrapper = None


def _get_scanpy_wrapper():
    """Load scanpy_wrapper from DE-ZILN/pkg (relative to this file)."""
    global _scanpy_wrapper
    if _scanpy_wrapper is None:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(this_dir, "..", "pkg", "scanpy_wrapper.py")
        spec = importlib.util.spec_from_file_location("scanpy_wrapper", file_path)
        _scanpy_wrapper = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_scanpy_wrapper)
    return _scanpy_wrapper


def tocsr_if_not_sparse(x):
    """Convert ndarray/dense matrix to CSR sparse format if not already sparse."""
    if sp.issparse(x):
        return x.copy()
    return sp.csr_matrix(x)


def prepare_adata_layers(adata):
    """
    Ensure adata has layers: counts, norm_counts, log1p_norm.
    Assumes adata.X is raw counts. Modifies adata in place.
    """
    if "counts" not in adata.layers:
        adata.layers["counts"] = tocsr_if_not_sparse(adata.X)
    if "norm_counts" not in adata.layers:
        temp = adata.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        adata.layers["norm_counts"] = tocsr_if_not_sparse(temp.X)
        del temp
    if "log1p_norm" not in adata.layers:
        temp = adata.copy()
        sc.pp.normalize_total(temp, target_sum=1e4)
        sc.pp.log1p(temp)
        adata.layers["log1p_norm"] = tocsr_if_not_sparse(temp.X)
        del temp
    return adata


def run_ln_de(
    adata,
    groupby,
    key_added="rank_genes_groups",
    layer="norm_counts",
    groups=None,
    reference="rest",
    sparse=True,
    **kwargs,
):
    """
    Run LN test. Uses pkg/scanpy_wrapper.rank_genes_groups_ln.
    Default reference='rest' gives one-vs-rest; set groups and reference for two-group.
    """
    sw = _get_scanpy_wrapper()
    sw.rank_genes_groups_ln(
        adata,
        groupby,
        key_added=key_added,
        layer=layer,
        groups=groups,
        reference=reference,
        sparse=sparse,
        **kwargs,
    )


def run_ttest_de(
    adata,
    groupby,
    key_added="rank_genes_groups",
    layer="log1p_norm",
    use_raw=False,
    groups=None,
    reference="rest",
    **kwargs,
):
    """Run Scanpy t-test on layer (default log1p_norm)."""
    _groups = "all" if groups is None else groups
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="t-test",
        key_added=key_added,
        layer=layer,
        use_raw=use_raw,
        groups=_groups,
        reference=reference,
        **kwargs,
    )


def run_wilcoxon_de(
    adata,
    groupby,
    key_added="rank_genes_groups",
    layer="log1p_norm",
    use_raw=False,
    groups=None,
    reference="rest",
    **kwargs,
):
    """Run Scanpy wilcoxon on layer (default log1p_norm)."""
    _groups = "all" if groups is None else groups
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method="wilcoxon",
        key_added=key_added,
        layer=layer,
        use_raw=use_raw,
        groups=_groups,
        reference=reference,
        **kwargs,
    )


def run_mast_de(
    adata,
    group_col,
    group_1=None,
    group_2=None,
    log1p_layer="log1p_norm",
    key_added="mast",
):
    """
    Run MAST differential expression via rpy2.

    - If group_1 and group_2 are provided: two-group comparison (group_1 vs group_2).
      Writes result to adata.uns[key_added] and returns it.
    - If both are None: one-vs-rest per cluster. Returns dict[cluster_str, DataFrame]
      with columns: names, logfoldchanges, scores, pvals, pvals_adj.

    Requires rpy2 and R package MAST.
    """
    if not _HAS_RPY2:
        raise ImportError("run_mast_de requires rpy2. Install with: pip install rpy2")
    importr("MAST")

    if log1p_layer not in adata.layers:
        raise ValueError(f"Layer '{log1p_layer}' not found in adata.layers")

    if group_1 is not None and group_2 is not None:
        # Two-group: subset cells and use their expression matrix
        mask = adata.obs[group_col].isin([group_1, group_2])
        ad_sub = adata[mask]
        X_sub = ad_sub.layers[log1p_layer]
        if sp.issparse(X_sub):
            X_sub = X_sub.toarray()
        cell_names_sub = list(ad_sub.obs_names)
        gene_names_sub = list(ad_sub.var_names)
        n_cells_sub, n_genes_sub = X_sub.shape
        X_T_sub = np.asfortranarray(X_sub.T)
        r_expr_sub = ro.r.matrix(
            ro.FloatVector(X_T_sub.ravel(order="F")),
            nrow=n_genes_sub,
            ncol=n_cells_sub,
            byrow=False,
        )
        r_expr_sub.rownames = ro.StrVector(gene_names_sub)
        r_expr_sub.colnames = ro.StrVector(cell_names_sub)
        ro.globalenv["expr_mat"] = r_expr_sub
        group_labels = ad_sub.obs[group_col].values
        binary_labels = np.where(group_labels == group_1, "target", "rest")
        cell_data = pd.DataFrame(
            {"wellKey": cell_names_sub, "group": binary_labels},
            index=cell_names_sub,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_cell_data = ro.conversion.py2rpy(cell_data)
        ro.globalenv["cell_data"] = r_cell_data

        ro.r("""
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
        """)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.globalenv["mast_result"])
        df = df.set_index("names")
        df = df.reindex(adata.var_names)
        result_dict = {
            "names": df.index.values,
            "logfoldchanges": df["logfoldchanges"].values,
            "scores": df["scores"].values,
            "pvals": df["pvals"].values,
            "pvals_adj": df["pvals_adj"].values,
        }
        de_key = f"{group_1}_vs_{group_2}"
        adata.uns[key_added] = {de_key: result_dict}
        ro.r("rm(expr_mat, cell_data, mast_result); gc()")
        return adata.uns[key_added]

    # One-vs-rest: loop over clusters (full adata)
    # R/MAST FromMatrix expects a dense matrix; we convert once and pass to R, then drop Python copies to limit memory
    X = adata.layers[log1p_layer]
    if sp.issparse(X):
        X = X.toarray()
    cell_names = list(adata.obs_names)
    gene_names = list(adata.var_names)
    n_cells, n_genes = X.shape
    X_T = np.asfortranarray(X.T)
    r_expr = ro.r.matrix(
        ro.FloatVector(X_T.ravel(order="F")),
        nrow=n_genes,
        ncol=n_cells,
        byrow=False,
    )
    r_expr.rownames = ro.StrVector(gene_names)
    r_expr.colnames = ro.StrVector(cell_names)
    ro.globalenv["expr_mat"] = r_expr
    del X, X_T  # free dense copies; R holds the data

    # One-vs-rest: loop over clusters
    cluster_labels = adata.obs[group_col].values
    unique_clusters = sorted(pd.unique(cluster_labels))
    results = {}
    for target_cluster in unique_clusters:
        binary_labels = np.where(cluster_labels == target_cluster, "target", "rest")
        cell_data = pd.DataFrame(
            {"wellKey": cell_names, "group": binary_labels},
            index=cell_names,
        )
        with localconverter(ro.default_converter + pandas2ri.converter):
            r_cell_data = ro.conversion.py2rpy(cell_data)
        ro.globalenv["cell_data"] = r_cell_data
        ro.r("""
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
        """)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df = ro.conversion.rpy2py(ro.globalenv["mast_result"])
        df = df.set_index("names")
        df = df.reindex(gene_names)
        df["names"] = df.index
        results[str(target_cluster)] = df[["names", "logfoldchanges", "scores", "pvals", "pvals_adj"]].copy()
    ro.r("rm(expr_mat, cell_data, mast_result); gc()")
    return results
