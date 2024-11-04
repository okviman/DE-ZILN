import numpy as np
import matplotlib.pyplot as plt

from utils import digamma, trigamma, get_intervals


def generate_count_data(r, p, n):
    x = np.random.negative_binomial(r, p, n)
    N_plus = np.sum(x > 0)
    N_0 = n - N_plus
    return x, N_plus, N_0


def estimate_ziln_mean(x, a, b):
    _, log_mean, _ = get_intervals(np.log(x[x>0]), a, b)
    return log_mean


def generate_scaled_binomial_draws(theta, n):
    rng = np.random.default_rng()
    N_plus = rng.binomial(n, theta)
    return N_plus / n


def generate_beta_draws(a, b):
    return np.random.beta(a, b)


def generate_ln_draws(a, b):
    mu = digamma(a) - digamma(a + b)
    sigma = np.sqrt(trigamma(a) - trigamma(a + b))
    return np.exp(np.random.normal(mu, sigma))


def visualize_fit_to_scaled_binomial():
    theta = 0.1
    n = 1000
    experiments = 10000
    scaled_binomials = np.zeros(experiments)
    betas = np.zeros_like(scaled_binomials)
    log_normals = np.zeros_like(scaled_binomials)
    normals = np.zeros_like(scaled_binomials)

    for i in range(experiments):
        scaled_binomials[i] = generate_scaled_binomial_draws(theta, n)
        betas[i] = generate_beta_draws(theta * n, n * (1 - theta))
        log_normals[i] = generate_ln_draws(theta * n, n * (1 - theta))
        normals[i] = np.random.normal(theta, np.sqrt(theta * (1 - theta) / n))

    fig, ax = plt.subplots(4, 1, sharex=True)
    ax[0].hist(scaled_binomials, bins=100)
    ax[1].hist(betas, bins=100, color='r')
    ax[2].hist(log_normals, bins=100)
    ax[3].hist(normals, bins=100)
    plt.tight_layout()
    plt.show()


def visualize_lfc_normality():
    r_treatment = 10
    r_ctrl = 20
    p = 0.01
    n = 1000
    experiments = 10000
    lfcs_ziln = np.zeros(experiments)
    for i in range(experiments):
        x_treatment, a_treatment, b_treatment = generate_count_data(r_treatment, p, n)
        x_ctrl, a_ctrl, b_ctrl = generate_count_data(r_ctrl, p, n)
        lt = estimate_ziln_mean(x_treatment, a_treatment, b_treatment)
        lc = estimate_ziln_mean(x_ctrl, a_ctrl, b_ctrl)
        lfcs_ziln[i] = lt - lc
    lfc = np.log(r_treatment * (1 - p) / p) - np.log(r_ctrl * (1 - p) / p)
    plt.hist(lfcs_ziln, bins=100)
    plt.vlines(lfc, 0, 300, color='r')
    plt.show()


def lfc_coverage():
    r_treatment = 1
    r_ctrl = 1
    p = 0.1
    x_treatment, a_treatment, b_treatment = generate_count_data(r_treatment, p, 1000)
    x_ctrl, a_ctrl, b_ctrl = generate_count_data(r_ctrl, p, 1000)
    f, ax = plt.subplots(3, 1)
    ax[0].set_title('Treatment NB')
    ax[0].hist(x_treatment, bins=100, label=f'$r={r_treatment},p={p}$')
    ax[0].legend(loc='upper right')
    ax[1].set_title('Ctrl NB')
    ax[1].hist(x_ctrl, bins=100, color='r', label=f'$r={r_ctrl},p={p}$')
    ax[1].legend(loc='upper right')

    coverage_ziln = []
    coverage_normal = []
    n_list = [10, 50, 100, 200, 500, 1000, 2000, 6000, 8000, 10000]
    for n in n_list:
        experiments = 2000
        lfc_ziln_intervals = np.zeros((2, experiments))
        lfc_normal_intervals = np.zeros((2, experiments))
        for i in range(experiments):
            x_treatment, a_treatment, b_treatment = generate_count_data(r_treatment, p, n)
            x_ctrl, a_ctrl, b_ctrl = generate_count_data(r_ctrl, p, n)

            for model in ['lognormal', 'naive']:

                _, lt, se_t = get_intervals(np.log(x_treatment[x_treatment > 0]), a_treatment, b_treatment, model=model)
                _, lc, se_c = get_intervals(np.log(x_ctrl[x_ctrl > 0]), a_ctrl, b_ctrl, model=model)

                estimated_lfc = lt - lc
                se_lfc = np.sqrt(se_t ** 2 + se_c ** 2)
                if model == 'lognormal':
                    lfc_ziln_intervals[:, i] = estimated_lfc + 1.96 * np.array([-se_lfc, se_lfc])
                else:
                    lfc_normal_intervals[:, i] = np.exp(estimated_lfc) + 1.96 * np.array([-se_lfc, se_lfc])

        lfc = np.log(r_treatment * (1 - p) / p) - np.log(r_ctrl * (1 - p) / p)

        print("n:", n)
        for model in ['lognormal', 'naive']:
            if model == 'lognormal':
                coverage_percentage = (np.sum(
                    (lfc > lfc_ziln_intervals[0, :]) * (lfc < lfc_ziln_intervals[1, :]))) / experiments
                coverage_ziln.append(coverage_percentage)
            else:
                coverage_percentage = (np.sum(
                    (np.exp(lfc) > lfc_normal_intervals[0, :]) * (np.exp(lfc) < lfc_normal_intervals[1, :]))) / experiments
                coverage_normal.append(coverage_percentage)
            print(model + ":", coverage_percentage)
    ax[2].plot(n_list, coverage_ziln, color='b', label='ZILN')
    ax[2].plot(n_list, coverage_normal, color='r', label='normal')
    ax[2].hlines(0.95, n_list[0], n_list[-1], color='black', linestyle='--', label='95% coverage')
    ax[2].legend(loc='best')
    ax[2].set_xlabel('$n$')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # visualize_lfc_normality()
    lfc_coverage()