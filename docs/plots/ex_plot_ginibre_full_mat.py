from beta_ensembles import GinibreEnsemble

ginibre = GinibreEnsemble() # beta must be 2 (default)
ginibre.sample_full_model(size_N=40)
ginibre.plot(normalization=True)