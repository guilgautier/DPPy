from dppy.beta_ensembles import HermiteEnsemble

hermite = HermiteEnsemble(beta=4) # beta must be in {1,2,4}, default beta=2
hermite.sample_full_model(size_N=500)
# hermite.plot(normalization=True)
hermite.hist(normalization=True)

# To compare with the sampling speed of the tridiagonal model simply use
# hermite.sample_banded_model(size_N=500)