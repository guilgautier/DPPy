from dppy.beta_ensembles.hermite import HermiteBetaEnsemble

hermite = HermiteBetaEnsemble(beta=4)  # beta in {0,1,2,4}, default beta=2
hermite.sample_full_model(size_N=500)
# hermite.plot(normalization=True)
hermite.hist(normalization=True)

# To compare with the sampling speed of the tridiagonal model simply use
# hermite.sample_banded_model(size_N=500)
