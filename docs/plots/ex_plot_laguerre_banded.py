from beta_ensembles import *

ensemble_name, beta = "laguerre", 4.15 # beta > 0
laguerre = BetaEnsemble(ensemble_name, beta=beta) # Create the laguerre object

# Parameters of the Gamma(shape, scale)/number of points
laguerre_params = {"shape":10000, "scale":2.0, "size":2000} 
# To match the full matrix model
# shape = 0.5*beta*(M-N+1)
# scale = 2.0
# size = N

mode = "banded"  # Banded (tridiagonal) matrix model

laguerre.sample(mode, **laguerre_params) # Sample

laguerre.hist(normalization=True) # Histogram of the eigenvalues/(beta*M)