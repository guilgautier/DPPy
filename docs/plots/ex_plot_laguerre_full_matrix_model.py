from dppy.beta_ensembles.laguerre import HermiteBetaEnsemble

laguerre = LaguerreBetaEnsemble(beta=1)  # beta in {0,1,2,4}, default beta=2
laguerre.sample_full_model(N=500, M=800)  # M >= N
# laguerre.plot(normalization=True)
laguerre.hist(normalization=True)

# To compare with the sampling speed of the tridiagonal model simply use
# laguerre.sample_banded_model(N=500, M=800)
