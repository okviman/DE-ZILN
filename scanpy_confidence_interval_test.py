import numpy as np
from utils import get_scanpy_lfcs
import matplotlib.pyplot as plt

"""
Scanpy function to get mean and vars used for computing the t statistic

def _get_mean_var(
    X: _SupportedArray, *, axis: Literal[0, 1] = 0
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if isinstance(X, sparse.spmatrix):
        mean, var = sparse_mean_variance_axis(X, axis=axis)
    else:
        mean = axis_mean(X, axis=axis, dtype=np.float64)
        mean_sq = axis_mean(elem_mul(X, X), axis=axis, dtype=np.float64)
        var = mean_sq - mean**2
    # enforce R convention (unbiased estimator) for variance
    var *= X.shape[axis] / (X.shape[axis] - 1)
    return mean, var
"""

np.random.seed(0)
nx = 1000
ny = 1000
n_genes = 1500
mu1 = 100 + np.abs(np.random.normal(15, 5, (1, n_genes))) * np.random.binomial(1, 0.1, (1, n_genes))
mu2 = 100
d1 = 0.2
d2 = 0.1

r1 = 1 / d1
r2 = 1 / d2
p1 = 1 / (1 + d1 * mu1)
p2 = 1 / (1 + d2 * mu2)

# Generate synthetic gene expression data
X = np.random.negative_binomial(n=r1, p=p1, size=(nx, n_genes))
Y = np.random.negative_binomial(n=r2, p=p2, size=(ny, n_genes))

true_lfcs = np.log2(mu2 / mu1)

sc_lfcs = get_scanpy_lfcs(X, Y, normalize=True)

# this is how Scanpy calculates the statistics used in the t test (based on their code)
# the results here were reproduced in the debugging console within the scanpy code, too
X = np.log(1e4 * X / (X.sum(1, keepdims=True)) + 1)
Y = np.log(1e4 * Y / (Y.sum(1, keepdims=True)) + 1)

mean_Y, mean_X = np.mean(Y, axis=0), np.mean(X, axis=0)

mean_sq_Y, mean_sq_X = np.mean(Y * Y, axis=0), np.mean(X * X, axis=0)
var_Y = (mean_sq_Y - mean_Y ** 2) * ny / (ny - 1)
var_X = (mean_sq_X - mean_X ** 2) * nx / (nx - 1)

se_Y = np.sqrt(var_Y / ny)
se_X = np.sqrt(var_X / nx)
se = np.sqrt(se_Y ** 2 + se_X ** 2) / np.log(2)

confidence_intervals = (mean_Y - mean_X) / np.log(2) + 1.96 * np.array([-se, se])
print(f"Estimated LFC is below CI {np.sum(sc_lfcs < confidence_intervals[0])} times")
print(f"Estimated LFC is above CI {np.sum(sc_lfcs > confidence_intervals[1])} times")
plt.errorbar(true_lfcs[0], (mean_Y - mean_X) / np.log(2), yerr=1.96 * se, fmt='o', label='Estimated CIs')
plt.plot(true_lfcs[0], true_lfcs[0], 'o', label='True LFCs')
plt.plot(true_lfcs[0], sc_lfcs, 'o', label='Estimated LFCs')
plt.xlabel('LFCs')
plt.ylabel('LFCs')
plt.title('Scanpy')
plt.legend()
plt.show()