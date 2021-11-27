import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import qr
from scipy.stats import binom, chisquare

from dppy.finite_dpps import FiniteDPP

r, N = 5, 10
e_vals = np.ones(r)
e_vecs, _ = qr(np.random.randn(N, r), mode="economic")

dpp_L = FiniteDPP("likelihood", projection=True, **{"L_eig_dec": (e_vals, e_vecs)})

nb_samples = 1000
dpp_L.flush_samples
for _ in range(nb_samples):
    dpp_L.sample_exact()

sizes = list(map(len, dpp_L.list_of_samples))

p = 0.5  # binomial parameter
rv = binom(r, p)

fig, ax = plt.subplots(1, 1)

x = np.arange(0, r + 1)

pdf = rv.pmf(x)
ax.plot(x, pdf, "ro", ms=8, label=r"pdf $Bin({}, {})$".format(r, p))

hist = np.histogram(sizes, bins=np.arange(0, r + 2), density=True)[0]
ax.vlines(x, 0, hist, colors="b", lw=5, alpha=0.5, label="hist of sizes")

ax.legend(loc="best", frameon=False)

plt.title("p_value = {:.3f}".format(chisquare(hist, pdf)[1]))
