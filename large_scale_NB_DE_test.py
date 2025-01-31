import numpy as np
import pandas as pd
import statsmodels.stats.multitest as smm
from utils import get_DELN_lfcs
from scanpy_ttest import get_test_results, scanpy_sig_test


np.random.seed(0)

nx = 1000
ny = 1000
n_genes = 1500
columns = ["method", "accuracy", "precision", "recall", "tpr", "tnr", "fpr", "fnr", "f1", "dispersion", "rep_no"]

# Create an empty DataFrame with these columns
df = pd.DataFrame(columns=columns)

# dispersions
d2 = 0.5
non_de_mu = 10
log_batch_factor = 0.
for d1 in [0.25, 0.5, 1.]:
    # with probability 0.1, there is Gaussian noise added to a gene in group 1
    rep_count = 20
    for rep in range(rep_count):
        z1 = np.random.binomial(1, 0.1, (1, n_genes))
        mu1 = non_de_mu + np.abs(np.random.normal(0, 5, (1, n_genes))) * z1
        mu2 = non_de_mu * np.exp(log_batch_factor)

        r1 = 1 / d1
        r2 = 1 / d2
        p1 = 1 / (1 + d1 * mu1)
        p2 = 1 / (1 + d2 * mu2)

        X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
        Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

        for method in ["DELN", "t-test", 't-test_overestim_var', 'wilcoxon']:
            if method == "DELN":
                _, DELN_p_vals = get_DELN_lfcs(Y, X, test='t')
                adj_pvals = smm.multipletests(DELN_p_vals, alpha=0.05, method='fdr_bh')[1]
            else:
                _, adj_pvals = scanpy_sig_test(X, Y, method=method)
                method = "Scanpy " + method

            results = get_test_results(adj_pvals, z1, verbose=False)
            results["method"] = method
            results["rep_no"] = rep
            results["dispersion"] = d1
            df = pd.concat([df, pd.DataFrame([results])], ignore_index=True)

df.to_csv(f"/Users/oskarkviman/Documents/"
          f"phd/DE-ZILN/simul/"
          f"test/NB_test_results/"
          f"nde_mu{int(non_de_mu)}/"
          f"d1_vs_d2_01_nde_mu_{int(non_de_mu)}.csv",
          index=False)

