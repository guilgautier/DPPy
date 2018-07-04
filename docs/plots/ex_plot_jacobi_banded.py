from beta_ensembles import *

ensemble_name, beta = "jacobi", 4.15 # beta > 0
jacobi = BetaEnsemble(ensemble_name, beta=beta) # Create the jacobi object

# Parameters of the Beta(a,b)/number of points
jacobi_params = {"a":300, "b":100, "size":500} 
# To match the full matrix model use
# a, b = 0.5*beta*(M_1-N+1), 0.5*beta*(M_2-N+1)
# size = N

sampling_mode = "banded" # Banded (tridiagonal) matrix model

jacobi.sample(sampling_mode, **jacobi_params) # Sample

jacobi.hist(normalization=True) # Histogram of the eigenvalues