from dppy.beta_ensembles import CircularEnsemble

circular = CircularEnsemble(beta=2) # beta must be >=0 integer, default beta=2

# See the cristallization of the configuration as beta increases
for b in (0, 1, 5, 10):

	circular.beta = b
	circular.sample_banded_model(size_N=30)
	circular.plot()