from beta_ensembles import *

ensemble_name, beta = "jacobi", 2 # beta = 1, 2, 4
jacobi = BetaEnsemble(ensemble_name, beta=beta) # Create the jacobi object

jacobi_params = {"M_1":1500, "M_2":1200, "N":1000} # Size of the matrices M_1/2xN (M_1/2>=N)
sampling_mode = "full" # Full matrix model

jacobi.sample(sampling_mode, **jacobi_params) # Sample

jacobi.hist(normalization=True) # Histogram of the eigenvalues