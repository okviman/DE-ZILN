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


def get_intervals(log_x, a, b, z=1.96, model='lognormal'):
    if model == 'naive':
        return interval_naive(log_x, b, z)
    n = log_x.size
    _, mu_bar, sigma_bar = intervals_ln(log_x, n, z)
    _, mu_log_beta, var_log_beta = intervals_beta(a, b, z)

    squared_standard_error_ln = np.var(log_x) / n + np.var(log_x) ** 2 / (2 * (n + 1))
    squared_standard_error_log_beta = var_log_beta
    se = np.sqrt(squared_standard_error_ln + squared_standard_error_log_beta)

    # estimate of the log of the mean of the ZILN
    log_mean_estimate = mu_bar + sigma_bar / 2 + mu_log_beta + var_log_beta / 2

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
