from __future__ import annotations
import numpy as np
import scipy.stats as stats
from scipy.stats.distributions import t
# from scipy.special import polygamma
# from typing import Optional, Sequence
# import math

from typing import Union, Iterable


class _TrigammaDiffCache:
    """
    prefix[k] = sum_{j=1..k} 1/j^2   (prefix[0] = 0)
    sum_{j=a..n-1} 1/j^2 = prefix[n-1] - prefix[a-1]
    """
    def __init__(self):
        self._prefix = [0.0]

    def ensure(self, max_k: int) -> None:
        cur = len(self._prefix) - 1
        if max_k <= cur:
            return

        s = self._prefix[-1]
        for j in range(cur + 1, max_k + 1):
            s += 1.0 / (j * j)
            self._prefix.append(s)


_cache = _TrigammaDiffCache()


def _to_positive_int_scalar(x) -> int:
    """
    Round to nearest integer and clamp to >= 1
    """
    xi = int(np.round(x))
    return max(1, xi)


def _to_positive_int_array(x):
    """
    Vectorized rounding and clamp to >= 1
    """
    xi = np.round(x).astype(int)
    xi[xi <= 0] = 1
    return xi


def trigamma_diff_int(
    a: Union[int, float, Iterable, np.ndarray],
    n: Union[int, float]
) -> Union[float, np.ndarray]:
    """
    Compute:
        psi_1(a) - psi_1(n) = sum_{j=a..n-1} 1/j^2

    - a can be scalar or array-like
    - values are rounded to nearest integer
    - all values are clamped to >= 1
    - cache is used only for array input
    """

    # ---- convert n ----
    n = _to_positive_int_scalar(n)

    # ---------- Scalar case ----------
    if np.isscalar(a):
        a = _to_positive_int_scalar(a)

        if a >= n:
            return 0.0

        s = 0.0
        for j in range(a, n):
            s += 1.0 / (j * j)
        return s

    # ---------- Array case ----------
    a_arr = _to_positive_int_array(np.asarray(a))

    # ensure cache up to n-1
    if n > 1:
        _cache.ensure(n - 1)

    result = np.zeros_like(a_arr, dtype=float)

    mask = (a_arr < n)

    if np.any(mask):
        prefix_n = _cache._prefix[n - 1]
        result[mask] = (
            prefix_n
            - np.array([_cache._prefix[x - 1] for x in a_arr[mask]])
        )

    return result

def digamma(x):
    return np.log(x) - 1 / (2 * x)

# def trigamma(x):
#         return 1 / x  + 0.5 / (x ** 2) +  1/(6.0*x**3)

def log_beta_param_estimates(a, b):
    mu = digamma(a) - digamma(a + b)
    #sigma_2 = trigamma(a) - trigamma(a + b)
    sigma_2 = trigamma_diff_int(a,a + b)
    return mu, sigma_2


def intervals_ln(log_x, n, z=1.96):
    mu_bar = np.mean(log_x)
    sigma_bar = np.var(log_x)

    se = np.sqrt(sigma_bar / n + sigma_bar ** 2 / (2 * (n - 1)))
    log_intervals = mu_bar + sigma_bar / 2 + z * np.array([-se, se])
    antilog_interval = np.exp(log_intervals)
    return antilog_interval, mu_bar, sigma_bar


def intervals_beta(a, b, z=1.96):
    mu_log_beta, var_log_beta = log_beta_param_estimates(a, b)
    se = np.sqrt(var_log_beta)
    log_intervals = mu_log_beta + var_log_beta / 2 + z * np.array([-se, se])
    antilog_interval = np.exp(log_intervals)
    return antilog_interval, mu_log_beta, var_log_beta


def get_intervals(log_x, a, b, z=1.96, model='lognormal', eps=0.):
    if model == 'naive':
        return interval_naive(log_x, b, z)
    n = log_x.size
    if n > 1:
        _, mu_bar, sigma_bar = intervals_ln(log_x, n, z)
        squared_standard_error_ln = sigma_bar / n + (sigma_bar ** 2) / (2 * (n - 1))
    else:
        # if there are only zero or one positive values, the mean estimate will be based on the log Beta mean
        mu_bar, sigma_bar = 0, 0
        squared_standard_error_ln = 0
    _, mu_log_beta, var_log_beta = intervals_beta(a + eps ** n, b, z)

    squared_standard_error_log_beta = var_log_beta
    se = np.sqrt(squared_standard_error_ln + squared_standard_error_log_beta)

    # estimate of the log of the mean of the ZILN
    a_hat = a + eps ** n
    log_mean_estimate = mu_bar + sigma_bar / 2 + np.log(a_hat / (a + b))
    # log_mean_estimate = mu_bar + sigma_bar / 2 + mu_log_beta + var_log_beta / 2

    log_intervals = log_mean_estimate + z * np.array([-se, se])
    antilog_interval = np.exp(log_intervals)
    return antilog_interval, log_mean_estimate, se

def interval_naive(log_x, N_0, z=1.96):
    zeros = np.zeros(N_0)
    data = np.concatenate([np.exp(log_x), zeros])
    n = data.size
    # sample mean
    mu_bar = np.mean(data)
    # use log as exp is used outside function
    log_mu_bar = np.log(mu_bar)
    sigma_bar = np.var(data)
    se = np.sqrt(sigma_bar / n)
    intervals = mu_bar + z * np.array([-se, se])
    return intervals, log_mu_bar, se


def get_intervals_synthetic_data(true_mu, true_sigma_2, true_theta, experiments=1000,
                                 n=500, z=1.96, model='lognormal', seed=0):
    # note: true_sigma_2 neq true variance, it's the var of the data-generating normal distribution
    np.random.seed(seed)
    intervals = np.zeros((2, experiments))
    estimated_means = np.zeros(experiments)
    for i in range(experiments):
        y = np.random.binomial(1, true_theta, n)
        N_plus = y.sum()
        N_0 = n - N_plus

        log_x = np.random.normal(true_mu, np.sqrt(true_sigma_2), y.sum())
        antilog_interval, log_mean_estimate, se = get_intervals(log_x, N_plus, N_0, z, model)
        intervals[:, i] = antilog_interval
        estimated_means[i] = np.exp(log_mean_estimate)
    return intervals, estimated_means


def get_LN_lfcs(Y_, X_, normalize=True, test='t', normalization='CP10K', return_standard_error=False, return_statistic=False):
    # Y is (n_cells, n_genes)

    G = Y_.shape[1]
    Y = Y_.astype(float).copy()
    X = X_.astype(float).copy()
    n = Y.shape[0]
    n_prime = X.shape[0]

    Y[Y <= 0] = np.nan  # Replace all non-positive with NaN
    n_plus = n - np.sum(np.isnan(Y), 0)
    all_zeros_Y = (n_plus == 0)

    X[X <= 0] = np.nan
    n_plus_prime = n_prime - np.sum(np.isnan(X), 0)
    all_zeros_X = (n_plus_prime == 0)


    if normalize and (normalization == 'CP10K'):
        X = 1e4 * X / np.nansum(X, 1, keepdims=True)
        Y = 1e4 * Y / np.nansum(Y, 1, keepdims=True)

    elif normalize and (normalization == 'median-of-ratios'):
        # the normalization scheme proposed in DESeq2
        denom_Y = np.exp(np.nanmean(np.log(Y), 0))
        denom_Y[np.isnan(denom_Y)] = 1  # Avoid division by NaN for unexpressed genes
        c_Y = np.nanmedian(Y / denom_Y, 1, keepdims=True)
        Y /= c_Y

        denom_X = np.exp(np.nanmean(np.log(X), 0))
        denom_X[np.isnan(denom_X)] = 1  # Avoid division by NaN for unexpressed genes
        c_X = np.nanmedian(X / denom_X, 1, keepdims=True)
        X /= c_X


    pos_mean_Y = np.ones(G, dtype=np.float32)  # to avoid NaNs in LFC when all counts are zero
    pos_mean_Y[~all_zeros_Y] = np.nanmean(Y[:, ~all_zeros_Y], axis=0)
    pos_mean_X = np.ones(G, dtype=np.float32)
    pos_mean_X[~all_zeros_X] = np.nanmean(X[:, ~all_zeros_X], axis=0)

    # \hat{a}
    a_hat_Y = np.ones(G, dtype=np.float32)  # to avoid NaNs in LFC when all counts are zero
    a_hat_Y[~all_zeros_Y] = n_plus[~all_zeros_Y]

    a_hat_X = np.ones(G, dtype=np.float32)
    a_hat_X[~all_zeros_X] = n_plus_prime[~all_zeros_X]

    # compute \log2\hat{theta} for each gene
    log2_theta_hat_Y = np.log2(a_hat_Y / n)
    log2_theta_hat_X = np.log2(a_hat_X / n_prime)

    # compute sample mean of positive counts
    log2_m_Y = np.log2(pos_mean_Y)
    log2_m_X = np.log2(pos_mean_X)

    lfc = (log2_theta_hat_Y + log2_m_Y) - (log2_theta_hat_X + log2_m_X)

    # compute standard errors
    #se_Y_1 = trigamma(a_hat_Y) - trigamma(n)
    se_Y_1 = trigamma_diff_int(a_hat_Y,int(n))
    se_Y_2 = np.ones(G, dtype=np.float32)  # to avoid NaNs in SE when all counts are zero
    se_Y_2[~all_zeros_Y] = np.log(1 + np.nanvar(Y[:, ~all_zeros_Y], axis=0) / (n_plus[~all_zeros_Y] * (2 ** log2_m_Y[~all_zeros_Y]) ** 2))
    se_Y = np.sqrt(se_Y_1 + se_Y_2) / np.log(2)

    #se_X_1 = trigamma(a_hat_X) - trigamma(n_prime)
    se_X_1 = trigamma_diff_int(a_hat_X,int(n_prime))
    se_X_2 = np.ones(G, dtype=np.float32)
    se_X_2[~all_zeros_X] = np.log(1 + np.nanvar(X[:, ~all_zeros_X], axis=0) / (n_plus_prime[~all_zeros_X] * (2 ** log2_m_X[~all_zeros_X]) ** 2))
    se_X = np.sqrt(se_X_1 + se_X_2) / np.log(2)

    if test == 't':
        statistic, p_vals = get_t_statistic(log2_theta_hat_Y + log2_m_Y, log2_theta_hat_X + log2_m_X,
                                            se_Y, se_X, n, n_prime)
    else:
        # z-test
        statistic, p_vals = compute_p_vals(log2_theta_hat_Y + log2_m_Y, log2_theta_hat_X + log2_m_X, se_Y, se_X)

    if return_standard_error:
        return lfc, p_vals, np.sqrt(se_X ** 2 + se_Y ** 2)
    
    if return_statistic:
        return lfc, p_vals, statistic

    return lfc, p_vals

import numpy as np

try:
    import scipy.sparse as sp
except ImportError:
    sp = None


def _ensure_sparse_positive(A):
    """
    Return CSR sparse matrix containing only strictly positive entries.
    (Mimics your `A[A <= 0] = np.nan` logic by *dropping* non-positives.)
    """
    if sp is None:
        raise ImportError("scipy.sparse is required for the sparse implementation.")

    if not sp.issparse(A):
        # If you ever pass dense here, this converts to sparse (can be expensive).
        A = np.asarray(A, dtype=np.float64)
        A[A <= 0] = 0.0
        A = sp.csr_matrix(A, dtype=np.float64)
        A.eliminate_zeros()
        return A

    A = A.tocsr().astype(np.float64, copy=True)
    if A.nnz:
        bad = A.data <= 0
        if np.any(bad):
            A.data[bad] = 0.0
            A.eliminate_zeros()
    return A


def _cp10k_sparse(A_csr):
    """Row-normalize sparse matrix to counts-per-10k. Preserves sparsity."""
    rs = np.asarray(A_csr.sum(axis=1)).ravel().astype(np.float64)
    scale = np.zeros_like(rs)
    nz = rs > 0
    scale[nz] = 1e4 / rs[nz]
    return sp.diags(scale).dot(A_csr)


def _geom_mean_nonzero_per_gene(A_csc):
    """Geometric mean per gene over nonzero entries only (vectorized via CSC indptr)."""
    G = A_csc.shape[1]
    indptr = A_csc.indptr
    counts = np.diff(indptr).astype(np.float64)

    denom = np.ones(G, dtype=np.float64)
    if A_csc.nnz == 0:
        return denom

    data = A_csc.data
    logdata = np.log(data)

    # sum of logs per column using reduceat
    sumlog = np.add.reduceat(logdata, indptr[:-1])
    mask = counts > 0
    denom[mask] = np.exp(sumlog[mask] / counts[mask])
    denom[~mask] = 1.0
    denom[~np.isfinite(denom)] = 1.0
    denom[denom == 0] = 1.0
    return denom


def _median_of_ratios_sparse(A_csr):
    """
    DESeq2 median-of-ratios, ignoring zeros.
    This is the only part that uses a per-row loop (still O(nnz)).
    """
    A_csc = A_csr.tocsc()
    denom = _geom_mean_nonzero_per_gene(A_csc)

    indptr = A_csr.indptr
    indices = A_csr.indices
    data = A_csr.data

    n = A_csr.shape[0]
    c = np.ones(n, dtype=np.float64)

    for i in range(n):
        s, e = indptr[i], indptr[i + 1]
        if s == e:
            c[i] = 1.0
            continue
        ratios = data[s:e] / denom[indices[s:e]]
        # guard against weird denom / inf
        ratios = ratios[np.isfinite(ratios) & (ratios > 0)]
        c[i] = np.median(ratios) if ratios.size else 1.0

    # divide each row by its size factor
    inv = np.ones_like(c)
    ok = c > 0
    inv[ok] = 1.0 / c[ok]
    return sp.diags(inv).dot(A_csr)


def _pos_mean_var_nnz_per_gene(A):
    """
    For strictly-positive sparse A (cells x genes), compute:
    n_plus[g]  = #positive entries (nnz per column)
    mean[g]    = mean of positive entries
    var[g]     = population variance of positive entries (ddof=0) to match np.nanvar default
    """
    A_csc = A.tocsc()
    n_plus = np.diff(A_csc.indptr).astype(np.float64)  # faster than getnnz(axis=0)
    s = np.asarray(A_csc.sum(axis=0)).ravel().astype(np.float64)
    ss = np.asarray(A_csc.multiply(A_csc).sum(axis=0)).ravel().astype(np.float64)

    mean = np.ones(A_csc.shape[1], dtype=np.float64)  # keep your "ones to avoid NaNs"
    var = np.zeros(A_csc.shape[1], dtype=np.float64)

    mask = n_plus > 0
    mean[mask] = s[mask] / n_plus[mask]
    # population var (ddof=0): E[x^2] - (E[x])^2
    ex2 = np.zeros_like(var)
    ex2[mask] = ss[mask] / n_plus[mask]
    var[mask] = ex2[mask] - mean[mask] ** 2
    var[var < 0] = 0.0  # numerical guard

    all_zeros = ~mask
    return n_plus, mean, var, all_zeros


def get_LN_lfcs_sparse(
    Y_,
    X_,
    normalize=True,
    test="t",
    normalization="CP10K",
    return_standard_error=False,
    return_statistic=False,
    eps=1e-12,
):
    """
    Sparse-friendly version of your LN LFC + test.

    Expects Y_, X_ to be scipy sparse (CSR/CSC) matrices (cells x genes).
    It treats *zeros and non-positives as absent* (like your NaN masking),
    and computes moments over strictly-positive entries only.
    """
    if sp is None:
        raise ImportError("scipy.sparse is required for sparse Y_/X_.")

    Y = _ensure_sparse_positive(Y_)
    X = _ensure_sparse_positive(X_)

    n = Y.shape[0]
    n_prime = X.shape[0]
    G = Y.shape[1]
    if X.shape[1] != G:
        raise ValueError("Y_ and X_ must have the same number of genes (columns).")

    # Normalization (keeps sparsity)
    if normalize and (normalization == "CP10K"):
        Y = _cp10k_sparse(Y)
        X = _cp10k_sparse(X)
    elif normalize and (normalization == "median-of-ratios"):
        # Works, but slower than CP10K due to per-row medians.
        Y = _median_of_ratios_sparse(Y)
        X = _median_of_ratios_sparse(X)

    # Positive-entry moments per gene
    n_plus, pos_mean_Y, var_Y, all_zeros_Y = _pos_mean_var_nnz_per_gene(Y)
    n_plus_prime, pos_mean_X, var_X, all_zeros_X = _pos_mean_var_nnz_per_gene(X)

    # \hat{a} (your fallback-to-ones behavior)
    a_hat_Y = np.ones(G, dtype=np.float64)
    a_hat_Y[~all_zeros_Y] = n_plus[~all_zeros_Y]
    a_hat_X = np.ones(G, dtype=np.float64)
    a_hat_X[~all_zeros_X] = n_plus_prime[~all_zeros_X]


    # log2 theta hats
    log2_theta_hat_Y = np.log2(a_hat_Y / float(n))
    log2_theta_hat_X = np.log2(a_hat_X / float(n_prime))

    # sample mean of positive counts
    # (pos_mean_* already has 1.0 for all-zeros genes, matching your behavior)
    log2_m_Y = np.log2(np.maximum(pos_mean_Y, eps))
    log2_m_X = np.log2(np.maximum(pos_mean_X, eps))

    se_Y_1 = trigamma_diff_int(a_hat_Y, int(n))
    se_X_1 = trigamma_diff_int(a_hat_X, int(n_prime))

    mu_Y = log2_theta_hat_Y + log2_m_Y
    mu_X = log2_theta_hat_X + log2_m_X
    lfc = mu_Y - mu_X

    se_Y_2 = np.ones(G, dtype=np.float64)
    se_X_2 = np.ones(G, dtype=np.float64)

    mY2 = np.maximum(pos_mean_Y, eps) ** 2
    mX2 = np.maximum(pos_mean_X, eps) ** 2

    maskY = ~all_zeros_Y
    maskX = ~all_zeros_X

    se_Y_2[maskY] = np.log(1.0 + var_Y[maskY] / (np.maximum(n_plus[maskY], 1.0) * mY2[maskY]))
    se_X_2[maskX] = np.log(1.0 + var_X[maskX] / (np.maximum(n_plus_prime[maskX], 1.0) * mX2[maskX]))

    se_Y = np.sqrt(np.maximum(se_Y_1 + se_Y_2, 0.0)) / np.log(2.0)
    se_X = np.sqrt(np.maximum(se_X_1 + se_X_2, 0.0)) / np.log(2.0)

    if test == "t":
        statistic, p_vals = get_t_statistic(mu_Y, mu_X, se_Y, se_X, n, n_prime)
    else:
        statistic, p_vals = compute_p_vals(mu_Y, mu_X, se_Y, se_X)

    if return_standard_error:
        return lfc, p_vals, np.sqrt(se_X**2 + se_Y**2)

    if return_statistic:
        return lfc, p_vals, statistic

    return lfc, p_vals



def get_seurat_lfcs(X, Y, normalize=True):
    # Manual calculation of the LFC based on how seurat implements it.
    # See Log fold-change calculation methods in https://www.biorxiv.org/content/10.1101/2022.05.09.490241v2.full.pdf
    if normalize:
        log_X = transform(X)
    else:
        log_X = np.log(X + 1)
    if normalize:
        log_Y = transform(Y)
    else:
        log_Y = np.log(Y + 1)

    return np.log2(np.mean(np.exp(log_Y) - 1, 0) + 1) - np.log2(np.mean(np.exp(log_X) - 1, 0) + 1)

def get_new_seurat_lfcs(X, Y, normalize=True, eps=1e-9):
    # Manual calculation of the LFC based on how seurat implements it.
    # See Log fold-change calculation methods in https://www.biorxiv.org/content/10.1101/2022.05.09.490241v2.full.pdf
    if normalize:
        log_X = transform(X)
    else:
        log_X = np.log(X + 1)
    if normalize:
        log_Y = transform(Y)
    else:
        log_Y = np.log(Y + 1)

    return np.log2((np.sum(np.exp(log_Y) - 1, 0) + eps) / Y.shape[0]) - np.log2((np.sum(np.exp(log_X) - 1, 0) + eps) / X.shape[0])


def get_scanpy_lfcs(X, Y, normalize=True):
    if normalize:
        log_X = transform(X)
    else:
        log_X = np.log(X + 1)
    if normalize:
        log_Y = transform(Y)
    else:
        log_Y = np.log(Y + 1)

    return np.log2(np.exp(np.mean(log_Y, 0)) - 1 + 1e-9) - np.log2(np.exp(np.mean(log_X, 0)) - 1 + 1e-9)


def transform(z):
    # log(10000 * z / z.sum(over genes for each cell) + 1)
    return np.log((z * 1e4 / z.sum(1, keepdims=True)) + 1)


def compute_p_vals(mean1, mean2, se1, se2):
    # Compute the test statistic
    z_stat = (mean1 - mean2) / ((se1 ** 2 + se2 ** 2) ** 0.5)

    # Compute the p-value for the two-tailed test
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return z_stat, p_value

def get_t_statistic(mean1, mean2, se1, se2, n1, n2):
    # implements two-sided t-test
    nu1, nu2 = n1 - 1, n2 - 1
    df = (se1 ** 2 + se2 ** 2) ** 2 / (se1 ** 4 / nu1 + se2 ** 4 / nu2)
    d = mean1 - mean2
    denom = ((se1 ** 2 + se2 ** 2) ** 0.5)
    t_statistic = d / denom
    t_dist = t(df)
    p_value = 2 * t_dist.sf(np.abs(t_statistic))
    return t_statistic, p_value

