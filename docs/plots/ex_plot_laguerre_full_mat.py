from dppy.beta_ensembles import LaguerreEnsemble

laguerre = LaguerreEnsemble(beta=1) # beta must be in {1,2,4}, default beta=2
laguerre.sample_full_model(size_N=500, size_M=800) # M >= N
# laguerre.plot(normalization=True)
laguerre.hist(normalization=True)

# To compare with the sampling speed of the tridiagonal model simply use
# laguerre.sample_banded_model(size_N=500, size_M=800)