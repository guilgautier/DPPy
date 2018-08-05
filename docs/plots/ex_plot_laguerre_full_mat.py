from beta_ensembles import *

ensemble_name, beta = "laguerre", 2 # beta = 1, 2, 4
laguerre = BetaEnsemble(ensemble_name, beta=beta) # Create the laguerre object

laguerre_params = {"M":1500, "N":1000} # Size of the matrix MxN (M>=N)
mode = "full" # Full matrix model

laguerre.sample(mode, **laguerre_params) # Sample

laguerre.hist(normalization=True) # Histogram of the eigenvalues/(beta*M)