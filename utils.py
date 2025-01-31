import numpy as np
import scipy.stats as stats
from scipy.stats.distributions import t

def digamma(x):
    return np.log(x) - 1 / (2 * x)


def trigamma(x):
    return 1 / x  # + 0.5 / (x ** 2)


def log_beta_param_estimates(a, b):
    mu = digamma(a) - digamma(a + b)
    sigma_2 = trigamma(a) - trigamma(a + b)
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


def get_ZILN_lfcs(X, Y, eps=0., return_p_vals=False):
    # Assuming X and Y are numpy arrays of raw counts
    # X: ctrl group (n_cells_x x n_genes)
    # Y: treatment group (n_cells_y x n_genes)

    n_genes = X.shape[-1]
    estimated_lfcs = np.zeros(n_genes)
    p_vals = np.zeros(n_genes)
    for g in range(n_genes):
        x = X[:, g]
        x_N_plus = np.sum(x > 0, axis=0)
        x_N_0 = np.sum(x == 0, axis=0)
        log_x = np.log(x[x > 0])

        y = Y[:, g]
        y_N_plus = np.sum(y > 0, axis=0)
        y_N_0 = np.sum(y == 0, axis=0)
        log_y = np.log(y[y > 0])

        if x_N_plus + y_N_plus == 0:
            # Convention 0 / 0 = 1
            estimated_lfc = 0.0
            p_vals[g] = 1e-90
        # elif (y_N_plus == 0) | (x_N_plus == 0):
        #     # LFC undefined
        #     estimated_lfc = np.inf
        #     p_vals[g] = 1e-90
        else:
            _, log_mu_x, se_x = get_intervals(log_x, x_N_plus, x_N_0, eps=eps ** x_N_plus)
            _, log_mu_y, se_y = get_intervals(log_y, y_N_plus, y_N_0, eps=eps ** y_N_plus)
            estimated_lfc = (log_mu_y - log_mu_x) / np.log(2)
            # no need to divide all terms by log(2) since they cancel in the z stat calculation
            p_vals[g] = compute_p_vals(log_mu_x, log_mu_y, se_x, se_y)

        estimated_lfcs[g] = estimated_lfc
    if return_p_vals:
        return estimated_lfcs, p_vals
    return estimated_lfcs


def get_DELN_lfcs(Y_, X_, normalize=True, test='t'):
    # Y is (n_cells, n_genes)
    eps = 1e-9

    Y = Y_.astype(float).copy()
    n = Y.shape[0]
    Y[Y <= 0] = np.nan  # Replace all non-positive with NaN
    n_plus = n - np.sum(np.isnan(Y), 0)

    X = X_.astype(float).copy()
    n_prime = X.shape[0]
    X[X <= 0] = np.nan
    n_plus_prime = n_prime - np.sum(np.isnan(X), 0)

    if normalize:

        X = 1e4 * X / np.nansum(X, 1, keepdims=True)
        Y = 1e4 * Y / np.nansum(Y, 1, keepdims=True)
        """
        denom_Y = np.exp(np.nanmean(np.log(Y), 0))
        c_Y = np.nanmedian(Y / denom_Y, 1, keepdims=True)
        Y /= c_Y
        Y /= np.nansum(denom_Y, 1, keepdims=True) 

        denom_X = np.exp(np.nanmean(np.log(X), 0))
        c_X = np.nanmedian(X / denom_X, 1, keepdims=True)
        X /= c_X
        """

    pos_mean_Y = np.nanmean(Y, axis=0)
    pos_mean_X = np.nanmean(X, axis=0)

    # \hat{a}
    a_hat_Y = n_plus + (eps ** (1 + n_plus))
    a_hat_X = n_plus_prime + (eps ** (1 + n_plus_prime))

    # compute \log2\hat{theta} for each gene
    log2_theta_hat_Y = np.log2(a_hat_Y / n)
    log2_theta_hat_X = np.log2(a_hat_X / n_prime)

    # compute sample mean of positive counts
    log2_m_Y = np.log2(pos_mean_Y)
    log2_m_X = np.log2(pos_mean_X)

    lfc = (log2_theta_hat_Y + log2_m_Y) - (log2_theta_hat_X + log2_m_X)

    # compute standard errors
    se_Y_1 = trigamma(a_hat_Y) - trigamma(n)
    se_Y_2 = np.log(1 + np.nanvar(Y, axis=0) / (n_plus * (2 ** log2_m_Y) ** 2))
    se_Y = np.sqrt(se_Y_1 + se_Y_2) / np.log(2)

    se_X_1 = trigamma(a_hat_X) - trigamma(n_prime)
    se_X_2 = np.log(1 + np.nanvar(X, axis=0) / (n_plus_prime * (2 ** log2_m_X) ** 2))
    se_X = np.sqrt(se_X_1 + se_X_2) / np.log(2)

    if test == 't':
        statistic, p_vals = get_t_statistic(log2_theta_hat_Y + log2_m_Y, log2_theta_hat_X + log2_m_X,
                                            se_Y, se_X, n, n_prime)
    else:
        # z-test
        statistic, p_vals = compute_p_vals(log2_theta_hat_Y + log2_m_Y, log2_theta_hat_X + log2_m_X, se_Y, se_X)

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

