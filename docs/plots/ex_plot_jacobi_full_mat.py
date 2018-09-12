from beta_ensembles import *

ensemble_name, beta = "jacobi", 2 # beta = 1, 2, 4
jacobi = BetaEnsemble(ensemble_name, beta=beta) # Create the jacobi object

jacobi_params = {"M_1":1500, "M_2":1200, "N":1000} # Size of the matrices NxM_1,M_2 (M_1,M_2>=N)
mode = "full" # Full matrix model

jacobi.sample(mode, **jacobi_params) # Sample

jacobi.hist(normalization=True) # Histogram of the eigenvalues