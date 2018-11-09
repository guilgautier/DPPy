from dppy.beta_ensembles import LaguerreEnsemble

laguerre = LaguerreEnsemble(beta=2.98) # beta must be in {1,2,4}, default beta=2
# Reference measure is Gamma(k, theta)
laguerre.sample_banded_model(shape=600, scale=2.0, size_N=400)
# laguerre.plot(normalization=True)
laguerre.hist(normalization=True)