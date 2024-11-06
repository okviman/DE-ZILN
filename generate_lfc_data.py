import numpy as np
import os


def generate_nb_counts(r1, r2, p1, p2, n_samples, n_experiments):
    counts_1 = np.random.negative_binomial(r1, p1, (n_samples, n_experiments))
    counts_2 = np.random.negative_binomial(r2, p2, (n_samples, n_experiments))
    lfc = np.log(r1 * (1 - p1) / p1) - np.log(r2 * (1 - p2) / p2)
    return counts_1, counts_2, lfc


def get_nb_lfc_data(path_to_save_datasets):
    n_list = [10, 20, 30, 50, 100, 1000, 2000]
    n_experiments = 1000

    r1 = 2
    r2 = 5
    p = 0.6

    ds_path = path_to_save_datasets + f"data_from_NB_parameters_r1_{r1}_r2_{r2}_p_0{int(10 * p)}"
    if not os.path.exists(ds_path):
        os.mkdir(ds_path)

    for n_samples in n_list:
        counts_1, counts_2, lfc = generate_nb_counts(r1, r2, p, p, n_samples, n_experiments)

        n_samples_ds_dir = ds_path + "/" + f"n_{n_samples}"
        os.mkdir(n_samples_ds_dir)

        np.save(n_samples_ds_dir + "/" + f"counts_1", counts_1)
        np.save(n_samples_ds_dir + "/" + f"counts_2", counts_2)
        np.save(n_samples_ds_dir + "/" + f"ground_truth_lfc", lfc)


if __name__ == '__main__':
    path = 'set path'
    get_nb_lfc_data(path)

