import numpy as np
import pandas as pd
import scanpy as sc
import statsmodels.stats.multitest as smm
from sklearn.metrics import confusion_matrix
from utils import get_LN_lfcs
import matplotlib.pyplot as plt
from scanpy_ttest import scanpy_sig_test, get_test_results


np.random.seed(0)
nx = 1000
ny = 1000
n_genes = 1500
base_mu = 50
mu1 = base_mu + np.zeros((1, n_genes))
mu2 = base_mu
true_lfcs = np.log2(mu2 / mu1)

plt.rcParams.update({'font.size': 50})
_, (sc_ax, LN_ax) = plt.subplots(1, 2, figsize=(30, 15), sharey=True)

d1_list = np.linspace(0.01, 1, 4)
d2_list = np.linspace(0.0001, 2, 20)

var_1 = d1_list * base_mu ** 2 + base_mu
var_2 = d2_list * base_mu ** 2 + base_mu
metric = "fpr"

for i, d1 in enumerate(d1_list):
    results = {"sc": [], "LN": []}
    for d2 in d2_list:
        r1 = 1 / d1
        r2 = 1 / d2
        p1 = 1 / (1 + d1 * mu1)
        p2 = 1 / (1 + d2 * mu2)

        # Generate synthetic gene expression data
        X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
        Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

        sc_lfs, sc_adj_pvals = scanpy_sig_test(X, Y)
        if np.sum(sc_adj_pvals >= 0.05) == n_genes:
            # 100% accuracy and 0% TPR --- conf_mat crashes in this setting
            sc_results = {"accuracy": 1., "fpr": 0.}
        else:
            sc_results = get_test_results(sc_adj_pvals, true_lfcs, verbose=False)

        LN_lfcs, LN_p_vals = get_LN_lfcs(Y, X, test='t')
        LN_adj_pvals = smm.multipletests(LN_p_vals, alpha=0.05, method='bonferroni')[1]
        if np.sum(LN_adj_pvals >= 0.05) == n_genes:
            # 100% accuracy and 0% TPR --- conf_mat crashes in this setting
            ln_results = {"accuracy": 1., "fpr": 0.}
        else:
            ln_results = get_test_results(LN_adj_pvals, true_lfcs, verbose=False)

        results["sc"] += [sc_results[metric]]
        results["LN"] += [ln_results[metric]]
    sc_ax.plot(var_2.astype(int), results["sc"], lw=5)
    LN_ax.plot(var_2.astype(int), results["LN"], label={f"Var$(X$)={int(var_1[i])}"}, lw=5)

sc_ax.set_xlabel(f'Var$(Y)$')
# sc_ax.legend(loc='best')
LN_ax.set_xlabel(f'Var$(Y)$')
LN_ax.set_title("LN's $t$-test")
sc_ax.set_title('$\log1$p $t$-test')
sc_ax.set_ylabel("FPR")
sc_ax.set_yticks(np.arange(0, 11) * 1 / 10)
LN_ax.legend(loc='best')
plt.show()



