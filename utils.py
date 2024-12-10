import numpy as np


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

    se = np.sqrt(sigma_bar / n + sigma_bar ** 2 / (2 * (n + 1)))
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
    if n > 0:
        _, mu_bar, sigma_bar = intervals_ln(log_x, n, z)
        squared_standard_error_ln = np.var(log_x) / n + np.var(log_x) ** 2 / (2 * (n + 1))
    else:
        # if there are no non-negative values, the LN is a point mass in 0 (exp(-inf) = 0)
        mu_bar, sigma_bar = -np.inf, 0
        squared_standard_error_ln = 0
    _, mu_log_beta, var_log_beta = intervals_beta(a + eps, b + eps, z)

    squared_standard_error_log_beta = var_log_beta
    se = np.sqrt(squared_standard_error_ln + squared_standard_error_log_beta)

    # estimate of the log of the mean of the ZILN
    log_mean_estimate = mu_bar + sigma_bar / 2 + mu_log_beta  # this term was incorrectly found here + var_log_beta / 2

    log_intervals = log_mean_estimate + z * np.array([-se, se])
    antilog_interval = np.exp(log_intervals)
    return antilog_interval, log_mean_estimate, se


def get_lfcs(X, Y):
    # Assuming X and Y are numpy arrays of raw counts
    # X: ctrl group (n_cells_x x n_genes)
    # Y: treatment group (n_cells_y x n_genes)
    n_genes = X.shape[-1]

    estimated_lfcs = np.zeros(n_genes)
    for g in range(n_genes):
        x = X[:, g]
        x_N_plus = np.sum(x > 0, axis=0)
        x_N_0 = np.sum(x == 0, axis=0)
        log_x = np.log(x[x > 0])
        _, log_mu_x, _ = get_intervals(log_x, x_N_plus, x_N_0, eps=1e-3)

        y = Y[:, g]
        y_N_plus = np.sum(y > 0, axis=0)
        y_N_0 = np.sum(y == 0, axis=0)
        log_y = np.log(y[y > 0])
        _, log_mu_y, _ = get_intervals(log_y, y_N_plus, y_N_0, eps=1e-3)

        estimated_lfc = (log_mu_y - log_mu_x) / np.log(2)
        estimated_lfcs[g] = estimated_lfc

    return estimated_lfcs



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
