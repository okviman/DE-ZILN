import numpy as np
import os
import pandas as pd


def generate_nb_counts(r1, r2, p1, p2, n_samples):
    counts_1 = np.random.negative_binomial(r1, p1, n_samples)
    counts_2 = np.random.negative_binomial(r2, p2, n_samples)
    return counts_1, counts_2


def get_nb_lfc_data(path_to_save_datasets):
    n_list = [10, 20, 30, 50, 100, 1000, 2000]
    n_genes = 1000

    r1 = 2
    r2 = 5
    p = 0.6

    ds_path = path_to_save_datasets + f"data_from_NB_parameters_r1_{r1}_r2_{r2}_p_0{int(10 * p)}"
    os.mkdir(ds_path)

    for n_samples in n_list:
        counts_1 = np.zeros((n_samples, n_genes))
        counts_2 = np.zeros_like(counts_1)
        for g in range(n_genes):
            if g % 10 == 0:
                # differentiated gene
                r_variable = r1
            else:
                # non-differentiated gene
                r_variable = r2
            counts_1[:, g], counts_2[:, g] = generate_nb_counts(r_variable, r2, p, p, n_samples)

        n_samples_ds_dir = ds_path + "/" + f"n_{n_samples}"
        os.mkdir(n_samples_ds_dir)

        df1 = pd.DataFrame(counts_1)
        df2 = pd.DataFrame(counts_2)
        df1.to_csv(n_samples_ds_dir + "/" + "treatment_counts.csv")
        df2.to_csv(n_samples_ds_dir + "/" + "control_counts.csv")


def get_random_lfc_data(path_to_save_datasets):
    n_list = [10, 20, 30, 50, 100, 1000, 2000]
    n_genes = 1000

    r2 = 5
    p = 0.6

    ds_path = path_to_save_datasets + f"data_from_NB_parameters_r2_{r2}_p_0{int(10 * p)}".replace('.', '')
    os.mkdir(ds_path)

    for n_samples in n_list:
        counts_1 = np.zeros((n_samples, n_genes))
        counts_2 = np.zeros_like(counts_1)
        lfcs = np.zeros(n_genes)
        for g in range(n_genes):
            if g % 10 == 0:
                # differentiated gene
                lfc = np.random.uniform(-4, 4)
                lfcs[g] = lfc
                # lfc = log(r1 * (1 - p) / p) - log(r2 * (1 - p) / p) = log(r1 / r2) --> r1 = exp(lfc + log(r2))
                r_variable = 2 ** (lfc + np.log2(r2))
            else:
                # non-differentiated gene
                r_variable = r2
            counts_1[:, g], counts_2[:, g] = generate_nb_counts(r_variable, r2, p, p, n_samples)

        n_samples_ds_dir = ds_path + "/" + f"n_{n_samples}"
        os.mkdir(n_samples_ds_dir)

        df1 = pd.DataFrame(counts_1)
        df2 = pd.DataFrame(counts_2)
        df3 = pd.DataFrame(lfcs)
        df1.to_csv(n_samples_ds_dir + "/" + "treatment_counts.csv")
        df2.to_csv(n_samples_ds_dir + "/" + "control_counts.csv")
        df3.to_csv(n_samples_ds_dir + "/" + "lfcs.csv")



if __name__ == '__main__':
    path = ''
    get_random_lfc_data(path)

