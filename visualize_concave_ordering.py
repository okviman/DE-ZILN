import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import nbinom

# -----------------------------------------------------------------------------
# 1. Define NB parameters so that both X and Y have the SAME MEAN (mu),
#    but Y has a LARGER variance => more dispersion => phi_Y > phi_X.
# -----------------------------------------------------------------------------

mu = 100.0

# We use the "number of successes" (r) and "success probability" (p) parameterization
# in scipy.stats.nbinom.  In that convention:
#    Mean = r*(1-p)/p
#    Var  = r*(1-p)/p^2
#
# If we fix mean = mu, then p = r/(r + mu).
# Smaller r => larger variance => more dispersion.

rX = 10  # larger r => less dispersion
rY = 4.  # smaller r => more dispersion

phiX = 1 / rX
phiY = 1 / rY

# Compute pX, pY so both have mean mu:
pX = rX / (rX + mu)
pY = rY / (rY + mu)

print("For X: r = {}, p = {:.3f} => mean = {:.1f}, var = {:.1f}".format(
    rX, pX, rX*(1-pX)/pX, rX*(1-pX)/(pX**2))
)
print("For Y: r = {}, p = {:.3f} => mean = {:.1f}, var = {:.1f}".format(
    rY, pY, rY*(1-pY)/pY, rY*(1-pY)/(pY**2))
)

# -----------------------------------------------------------------------------
# 2. Compute and plot the PMFs for X and Y up to some max count 'k_max'.
# -----------------------------------------------------------------------------

k_max = mu + 2 * np.sqrt((mu + mu ** 2 * 1 / rY))
k_vals = np.arange(k_max+1)

# PMFs
pmf_X = nbinom.pmf(k_vals, rX, pX)
pmf_Y = nbinom.pmf(k_vals, rY, pY)

# -----------------------------------------------------------------------------
# 3. Compute the sign of pY(k) - pX(k)
# -----------------------------------------------------------------------------
diff = pmf_Y - pmf_X
sign_diff = np.sign(diff)  # -1 if (pY < pX), 0 if equal, +1 if (pY > pX)

# -----------------------------------------------------------------------------
# 4. Plot:
#    - Top subplot: the two PMFs
#    - Bottom subplot: the sign of the difference as a step function
# -----------------------------------------------------------------------------

plt.rcParams.update({'font.size': 30})
fig, axes = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

# Top: PMFs
axes[0].plot(k_vals, pmf_X, 'bo-', label=f'$p_X$ = NB($\mu$={mu:.0f}, $\phi$={phiX:.1f})')
axes[0].plot(k_vals, pmf_Y, 'ro-', label=f'$p_Y$ = NB($\mu$={mu:.0f}, $\phi$={phiY:.1f})')
axes[0].set_ylabel("PMF")
axes[0].legend()
axes[0].set_title("Negative Binomial Distributions with Same Mean, Different Dispersion")

# Bottom: Sign of the difference
axes[1].step(k_vals, sign_diff, where='mid', color='green')
axes[1].set_ylim([-1.2, 1.2])
axes[1].set_yticks([-1, 0, 1])
axes[1].set_yticklabels(['-1','0','+1'])
axes[1].set_ylabel("sign($p_Y$ - $p_X$)")
axes[1].set_xlabel("$u$")

plt.tight_layout()
plt.show()
