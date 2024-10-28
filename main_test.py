import numpy as np
from utils import get_intervals_synthetic_data


if __name__ == '__main__':
    mu = 0
    sigma_2 = 1.5
    theta = 0.1
    n = 1000
    n_experiments = 2000

    true_mean = theta * np.exp(mu + 0.5 * sigma_2)

    # LN
    intervals, estimated_means = get_intervals_synthetic_data(
        mu, sigma_2, theta, experiments=n_experiments, n=n, model='lognormal'
    )
    print("LN RESULTS:")
    coverage_percentage = (np.sum((true_mean > intervals[0, :]) * (true_mean < intervals[1, :]))) / n_experiments
    print("Coverage percentage (ideal = 0.95): ", coverage_percentage)
    print("RMSE to the true mean: ", np.sqrt(np.mean((true_mean - estimated_means) ** 2)))

    # NAIVE (normal)
    intervals, estimated_means = get_intervals_synthetic_data(
        mu, sigma_2, theta, experiments=n_experiments, n=n, model='naive'
    )
    print("NAIVE RESULTS:")
    coverage_percentage = (np.sum((true_mean > intervals[0, :]) * (true_mean < intervals[1, :]))) / n_experiments
    print("Coverage percentage (ideal = 0.95): ", coverage_percentage)
    print("RMSE to the true mean: ", np.sqrt(np.mean((true_mean - estimated_means) ** 2)))


