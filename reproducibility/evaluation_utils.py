"""
Shared evaluation metrics for DE results: Jaccard, AUPR, GSEA, precision@k, avg |LFC|, n_sig_genes.

Works with AnnData (rank_genes_groups) and DataFrame (e.g. MAST results with names, scores, pvals_adj).
"""

from __future__ import annotations

import tempfile
from typing import Any, Dict, List, Optional, Set, Union

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import average_precision_score


# ---- Low-level: set-based and ranking-based ----

def jaccard_index(set_a: Union[Set[str], List[str]], set_b: Union[Set[str], List[str]]) -> float:
    """Jaccard index between two gene sets. Returns 0 if both empty."""
    a = set(str(x).strip() for x in set_a)
    b = set(str(x).strip() for x in set_b)
    union = len(a | b)
    if union == 0:
        return 0.0
    return len(a & b) / union


def aupr_from_ranking(
    gene_list: List[str],
    score_list: np.ndarray,
    ground_truth: Union[Set[str], List[str]],
    order_only: bool = False,
) -> float:
    """
    AUPR (area under precision-recall) with ground_truth as positives.
    Genes are ranked by score descending (higher score = rank 1).
    If order_only=True, scores are replaced by ranks (n, n-1, ..., 1) so only order matters.
    """
    gt_set = {str(g).strip() for g in ground_truth}
    genes = [str(g).strip() for g in gene_list]
    scores = np.asarray(score_list, dtype=float)
    # Sort by score descending (higher = rank 1)
    order = np.argsort(-scores)
    genes = [genes[i] for i in order]
    scores = scores[order]
    if order_only:
        n = len(genes)
        scores = np.arange(n, 0, -1, dtype=float)
    y_true = np.array([1 if g in gt_set else 0 for g in genes], dtype=int)
    if y_true.sum() == 0 or y_true.sum() == len(y_true):
        return float("nan")
    mask = ~np.isnan(scores)
    if not np.any(mask):
        return float("nan")
    y_true = y_true[mask]
    y_score = scores[mask]
    return float(average_precision_score(y_true, y_score))


def precision_at_k(ranked_genes: List[str], marker_genes: Union[Set[str], List[str]], k: int) -> float:
    """Precision@k = (# markers in top k) / k. Returns 0 if k <= 0."""
    if k <= 0:
        return 0.0
    markers = {str(m).strip() for m in marker_genes}
    top_k = ranked_genes[:k]
    return sum(1 for g in top_k if str(g).strip() in markers) / k


# ---- From AnnData (rank_genes_groups) ----

def _get_rank_genes_df(adata, group: str, key: str) -> pd.DataFrame:
    """Get DE result as DataFrame for one group. Columns: names, scores, logfoldchanges, pvals_adj."""
    return sc.get.rank_genes_groups_df(adata, group=group, key=key)


def get_top_sig_genes_adata(
    adata,
    group: str,
    top_n: int,
    key: str = "rank_genes_groups",
    pval_threshold: float = 0.05,
) -> Set[str]:
    """Top N significant genes for group (sorted by score descending)."""
    df = _get_rank_genes_df(adata, group, key)
    sig = df[df["pvals_adj"] < pval_threshold]
    if sig.shape[0] == 0:
        return set()
    sig = sig.sort_values("scores", ascending=False)
    return set(sig["names"].head(top_n).astype(str).tolist())


def get_all_sig_genes_adata(
    adata,
    group: str,
    key: str = "rank_genes_groups",
    pval_threshold: float = 0.05,
    positive_score_only: bool = False,
) -> Set[str]:
    """All significant genes for group. If positive_score_only, keep only genes with score > 0 (upregulated)."""
    df = _get_rank_genes_df(adata, group, key)
    sig = df[df["pvals_adj"] < pval_threshold]
    if positive_score_only and "scores" in sig.columns:
        sig = sig[sig["scores"] > 0]
    return set(sig["names"].astype(str).tolist())


def jaccard_top_adata(
    adata,
    group: str,
    marker_genes: Union[Set[str], List[str]],
    top_n: int,
    key: str = "rank_genes_groups",
    pval_threshold: float = 0.05,
) -> float:
    """Jaccard index between top N sig genes and marker set."""
    top = get_top_sig_genes_adata(adata, group, top_n, key=key, pval_threshold=pval_threshold)
    return jaccard_index(top, marker_genes)


def jaccard_all_sig_adata(
    adata,
    group: str,
    marker_genes: Union[Set[str], List[str]],
    key: str = "rank_genes_groups",
    pval_threshold: float = 0.05,
    positive_score_only: bool = True,
) -> float:
    """Jaccard index between sig genes (optionally positive score only) and marker set."""
    sig = get_all_sig_genes_adata(
        adata, group, key=key, pval_threshold=pval_threshold, positive_score_only=positive_score_only
    )
    return jaccard_index(sig, marker_genes)


def aupr_adata(
    adata,
    group: str,
    marker_genes: Union[Set[str], List[str]],
    key: str = "rank_genes_groups",
    order_only: bool = False,
) -> float:
    """AUPR with markers as ground truth. Ranks by score descending. If order_only=True, only rank order is used."""
    df = _get_rank_genes_df(adata, group, key)
    if df.empty or "scores" not in df.columns:
        return float("nan")
    df = df.dropna(subset=["scores"]).sort_values("scores", ascending=False)
    genes = df["names"].astype(str).tolist()
    scores = df["scores"].values
    return aupr_from_ranking(genes, scores, marker_genes, order_only=order_only)


def precision_at_k_adata(
    adata,
    group: str,
    marker_genes: Union[Set[str], List[str]],
    k_values: List[int],
    key: str = "rank_genes_groups",
) -> Dict[int, float]:
    """Precision@k for each k in k_values (genes ranked by score descending)."""
    df = _get_rank_genes_df(adata, group, key)
    df = df.sort_values("scores", ascending=False)
    ranked = df["names"].astype(str).tolist()
    return {k: precision_at_k(ranked, marker_genes, k) for k in k_values}


def avg_abs_lfc_adata(
    adata,
    group: str,
    key: str = "rank_genes_groups",
    pval_threshold: float = 0.05,
) -> float:
    """Mean absolute log fold change over significant genes."""
    df = _get_rank_genes_df(adata, group, key)
    sig = df[df["pvals_adj"] < pval_threshold]
    if sig.empty or "logfoldchanges" not in sig.columns:
        return float("nan")
    return float(sig["logfoldchanges"].abs().mean())


def n_sig_genes_adata(
    adata,
    group: str,
    key: str = "rank_genes_groups",
    pval_threshold: float = 0.05,
    positive_score_only: bool = False,
) -> int:
    """Number of significant genes. If positive_score_only, count only those with score > 0 (upregulated)."""
    df = _get_rank_genes_df(adata, group, key)
    mask = df["pvals_adj"] < pval_threshold
    if positive_score_only and "scores" in df.columns:
        mask = mask & (df["scores"] > 0)
    return int(mask.sum())


def n_genes_de_table_adata(
    adata,
    group: str,
    key: str = "rank_genes_groups",
) -> int:
    """Number of genes in the DE table for this group (total rows)."""
    df = _get_rank_genes_df(adata, group, key)
    return 0 if df.empty else len(df)


def frac_de_genes_valid_score_adata(
    adata,
    group: str,
    key: str = "rank_genes_groups",
    scores_col: str = "scores",
) -> float:
    """Fraction of genes in the DE table that have a non-NaN score (usable for AUPR/GSEA)."""
    df = _get_rank_genes_df(adata, group, key)
    if df.empty or scores_col not in df.columns:
        return float("nan")
    n = len(df)
    n_valid = df[scores_col].notna().sum()
    return float(n_valid / n) if n > 0 else float("nan")


# ---- From DataFrame (e.g. MAST: names, scores, pvals_adj, logfoldchanges) ----

def get_top_sig_genes_df(
    df: Optional[pd.DataFrame],
    top_n: int,
    pval_threshold: float = 0.05,
    names_col: str = "names",
    scores_col: str = "scores",
    pval_col: str = "pvals_adj",
) -> Set[str]:
    """Top N significant genes from a DE DataFrame."""
    if df is None or len(df) == 0:
        return set()
    sig = df[df[pval_col] < pval_threshold]
    if len(sig) == 0:
        return set()
    sig = sig.sort_values(scores_col, ascending=False)
    return set(sig[names_col].head(top_n).astype(str).tolist())


def get_all_sig_genes_df(
    df: Optional[pd.DataFrame],
    pval_threshold: float = 0.05,
    pval_col: str = "pvals_adj",
    names_col: str = "names",
    scores_col: str = "scores",
    positive_score_only: bool = False,
) -> Set[str]:
    """All significant genes from a DE DataFrame. If positive_score_only, keep only score > 0."""
    if df is None or len(df) == 0:
        return set()
    sig = df[df[pval_col] < pval_threshold]
    if positive_score_only and scores_col in sig.columns:
        sig = sig[sig[scores_col] > 0]
    return set(sig[names_col].astype(str).tolist())


def jaccard_top_df(
    df: Optional[pd.DataFrame],
    marker_genes: Union[Set[str], List[str]],
    top_n: int,
    pval_threshold: float = 0.05,
) -> float:
    """Jaccard between top N sig genes and marker set."""
    top = get_top_sig_genes_df(df, top_n, pval_threshold=pval_threshold)
    return jaccard_index(top, marker_genes)


def jaccard_all_sig_df(
    df: Optional[pd.DataFrame],
    marker_genes: Union[Set[str], List[str]],
    pval_threshold: float = 0.05,
    positive_score_only: bool = True,
) -> float:
    """Jaccard between sig genes (optionally positive score only) and marker set."""
    sig = get_all_sig_genes_df(
        df, pval_threshold=pval_threshold, positive_score_only=positive_score_only
    )
    return jaccard_index(sig, marker_genes)


def aupr_df(
    df: Optional[pd.DataFrame],
    marker_genes: Union[Set[str], List[str]],
    names_col: str = "names",
    scores_col: str = "scores",
    order_only: bool = False,
) -> float:
    """AUPR from DataFrame (markers = ground truth). Ranks by score descending. If order_only=True, only rank order is used."""
    if df is None or len(df) == 0:
        return float("nan")
    df = df.dropna(subset=[scores_col]).sort_values(scores_col, ascending=False)
    if len(df) == 0:
        return float("nan")
    genes = df[names_col].astype(str).tolist()
    scores = df[scores_col].values
    return aupr_from_ranking(genes, scores, marker_genes, order_only=order_only)


def precision_at_k_df(
    df: Optional[pd.DataFrame],
    marker_genes: Union[Set[str], List[str]],
    k_values: List[int],
    scores_col: str = "scores",
    names_col: str = "names",
) -> Dict[int, float]:
    """Precision@k from DataFrame (ranked by score descending)."""
    if df is None or len(df) == 0:
        return {k: 0.0 for k in k_values}
    df = df.sort_values(scores_col, ascending=False)
    ranked = df[names_col].astype(str).tolist()
    return {k: precision_at_k(ranked, marker_genes, k) for k in k_values}


def avg_abs_lfc_df(
    df: Optional[pd.DataFrame],
    pval_threshold: float = 0.05,
    pval_col: str = "pvals_adj",
    lfc_col: str = "logfoldchanges",
) -> float:
    """Mean absolute LFC over significant genes."""
    if df is None or len(df) == 0:
        return float("nan")
    sig = df[df[pval_col] < pval_threshold]
    if len(sig) == 0 or lfc_col not in sig.columns:
        return float("nan")
    return float(sig[lfc_col].abs().mean())


def n_sig_genes_df(
    df: Optional[pd.DataFrame],
    pval_threshold: float = 0.05,
    pval_col: str = "pvals_adj",
    scores_col: str = "scores",
    positive_score_only: bool = False,
) -> int:
    """Number of significant genes. If positive_score_only, count only score > 0."""
    if df is None or len(df) == 0:
        return 0
    mask = df[pval_col] < pval_threshold
    if positive_score_only and scores_col in df.columns:
        mask = mask & (df[scores_col] > 0)
    return int(mask.sum())


def n_genes_de_table_df(
    df: Optional[pd.DataFrame],
) -> int:
    """Number of genes in the DE table (total rows). Returns 0 if df is None or empty."""
    if df is None or len(df) == 0:
        return 0
    return len(df)


def frac_de_genes_valid_score_df(
    df: Optional[pd.DataFrame],
    scores_col: str = "scores",
) -> float:
    """Fraction of genes in the DE table that have a non-NaN score (usable for AUPR/GSEA)."""
    if df is None or len(df) == 0 or scores_col not in df.columns:
        return float("nan")
    n = len(df)
    n_valid = df[scores_col].notna().sum()
    return float(n_valid / n) if n > 0 else float("nan")


# ---- GSEA (optional, requires gseapy) ----

def run_gsea_prerank(
    gene_list: List[str],
    score_list: np.ndarray,
    ground_truth: Union[Set[str], List[str]],
    return_log10_p: bool = False,
    order_only: bool = False,
) -> Dict[str, Any]:
    """
    GSEA pre-rank with ground truth as the single gene set.
    Returns dict with at least "pvalue"; optionally "log10_pvalue", "nes".
    If order_only=True, scores are replaced by ranks (n, n-1, ..., 1) so only order matters.
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError("GSEA requires gseapy. Install with: pip install gseapy") from None

    gt_set = {str(g).strip() for g in ground_truth}
    genes_list = []
    scores_list = []
    for g, s in zip(gene_list, score_list):
        if np.isfinite(s):
            genes_list.append(str(g).strip())
            scores_list.append(float(s))
    if order_only and len(scores_list) > 0:
        # Replace with rank-based: rank 1 -> n, rank n -> 1
        n = len(scores_list)
        scores_list = list(np.arange(n, 0, -1, dtype=float))
    if len(genes_list) == 0:
        out = {"pvalue": float("nan")}
        if return_log10_p:
            out["log10_pvalue"] = float("nan")
        return out
    ranked_set = set(genes_list)
    ranked_upper_to_orig = {g.upper(): g for g in genes_list}
    gene_set_filtered = [ranked_upper_to_orig[m.upper()] for m in gt_set if m.upper() in ranked_upper_to_orig]
    if len(gene_set_filtered) == 0:
        out = {"pvalue": float("nan")}
        if return_log10_p:
            out["log10_pvalue"] = float("nan")
        return out

    rnk = pd.Series(np.asarray(scores_list, dtype=np.float64), index=genes_list)
    gene_sets = {"ground_truth": gene_set_filtered}
    import contextlib
    import io
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                res = gp.prerank(
                    rnk=rnk,
                    gene_sets=gene_sets,
                    processes=1,
                    permutation_num=1000,
                    outdir=tmpdir,
                    format="png",
                    seed=42,
                    verbose=False,
                    no_plot=True,
                    min_size=1,
                    max_size=5000,
                )
        except Exception as e:
            raise RuntimeError(f"GSEA prerank failed: {e}") from e
        out = {}
        if res.res2d is not None and len(res.res2d) > 0:
            row = res.res2d.iloc[0]
            pval = row.get("NOM p-val", row.get("FDR q-val", float("nan")))
            pval = float(pval) if pd.notna(pval) else float("nan")
            out["pvalue"] = pval
            if return_log10_p:
                out["log10_pvalue"] = np.log10(max(pval, 1e-300)) if np.isfinite(pval) and pval > 0 else float("nan")
            out["nes"] = float(res.res2d.iloc[0]["NES"]) if "NES" in res.res2d.columns else float("nan")
        else:
            out["pvalue"] = float("nan")
            if return_log10_p:
                out["log10_pvalue"] = float("nan")
            if "nes" not in out:
                out["nes"] = float("nan")
    return out


def gsea_nes_adata(
    adata,
    group: str,
    marker_genes: Union[Set[str], List[str]],
    key: str = "rank_genes_groups",
    order_only: bool = False,
) -> float:
    """GSEA NES with markers as gene set. Only genes with positive score (upregulated) are used as prerank input. If order_only=True, only rank order is used."""
    df = _get_rank_genes_df(adata, group, key)
    if df.empty or "scores" not in df.columns:
        return float("nan")
    df = df.dropna(subset=["scores"]).sort_values("scores", ascending=False)
    df = df[df["scores"] > 0]
    if df.empty:
        return float("nan")
    gene_list = df["names"].astype(str).tolist()
    score_list = df["scores"].values
    try:
        out = run_gsea_prerank(gene_list, score_list, marker_genes, order_only=order_only)
        return float(out.get("nes", float("nan")))
    except Exception:
        return float("nan")


def gsea_nes_df(
    df: Optional[pd.DataFrame],
    marker_genes: Union[Set[str], List[str]],
    names_col: str = "names",
    scores_col: str = "scores",
    order_only: bool = False,
) -> float:
    """GSEA NES from DataFrame. Only genes with positive score (upregulated) are used as prerank input. If order_only=True, only rank order is used."""
    if df is None or len(df) == 0:
        return float("nan")
    df = df.dropna(subset=[scores_col]).sort_values(scores_col, ascending=False)
    df = df[df[scores_col] > 0]
    if len(df) == 0:
        return float("nan")
    gene_list = df[names_col].astype(str).tolist()
    score_list = df[scores_col].values
    try:
        out = run_gsea_prerank(gene_list, score_list, marker_genes, order_only=order_only)
        return float(out.get("nes", float("nan")))
    except Exception:
        return float("nan")
