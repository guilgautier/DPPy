from finite_dpps import *

r, N = 4, 10
A = np.random.randn(r, N)

eig_vecs, _ = la.qr(A.T, mode="economic")
eig_vals = np.random.rand(r) # 0< <1
K = (eig_vecs*eig_vals).dot(eig_vecs.T)

DPP = FiniteDPP("inclusion", **{"K":K})

print(DPP)
# DPP defined through inclusion kernel
# Parametrized by dict_keys(['K'])
# - sampling mode = None
# - number of samples = 0

DPP.plot_kernel()