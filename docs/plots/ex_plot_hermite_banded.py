from beta_ensembles import *

ensemble_name, beta = "hermite", 4.15 # beta >0
hermite = BetaEnsemble(ensemble_name, beta=beta) # Create the hermite object

# Parameters of the Gaussian/number of points
hermite_params = {"loc":0.0, "scale":np.sqrt(2), "size":1000} 
# To match the full matrix model use
# loc = 0.0
# scale = np.sqrt(2)
# size = N

sampling_mode = "banded" # Banded (tridiagonal) matrix model

hermite.sample(sampling_mode, **hermite_params) # Sample

hermite.hist(normalization=True) # Histogram of the eigenvalues/sqrt(beta*size)