# from numpy import sqrt
from numpy.random import rand, randn
from scipy.linalg import qr

from dppy.finite.dpp import FiniteDPP

r, N = 4, 10
e_vecs, _ = qr(randn(N, r), mode="economic")

# Inclusion K
e_vals_K = rand(r)  # in [0, 1]
dpp_K = FiniteDPP("correlation", **{"K_eig_dec": (e_vals_K, e_vecs)})
# or
# K = (e_vecs * e_vals_K).dot(e_vecs.T)
# dpp_K = FiniteDPP('correlation', **{'K': K})
dpp_K.plot_kernel()

# Marginal L
e_vals_L = e_vals_K / (1.0 - e_vals_K)
dpp_L = FiniteDPP("likelihood", **{"L_eig_dec": (e_vals_L, e_vecs)})
# or
# L = (e_vecs * e_vals_L).dot(e_vecs.T)
# dpp_L = FiniteDPP('likelihood', **{'L': K})
# Phi = (e_vecs * sqrt(e_vals_L)).T
# dpp_L = FiniteDPP('likelihood', **{'L_gram_factor': Phi})  # L = Phi.T Phi
dpp_L.plot_kernel()
