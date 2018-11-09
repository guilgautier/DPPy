from dppy.beta_ensembles import HermiteEnsemble

hermite = HermiteEnsemble(beta=5.43)
# Reference measure is N(mu, sigma^2)
hermite.sample_banded_model(loc=0.0, scale=1.0, size_N=500)
# hermite.plot(normalization=True)
hermite.hist(normalization=True)