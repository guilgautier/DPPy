from dppy.beta_ensembles.hermite import HermiteBetaEnsemble

hermite = HermiteBetaEnsemble(beta=5.43)  # beta can be >=0, default beta=2
# Reference measure is N(mu, sigma^2)
hermite.sample_banded_model(loc=0.0, scale=1.0, N=500)
# hermite.plot(normalization=True)
hermite.hist(normalization=True)
