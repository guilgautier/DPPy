from beta_ensembles import *

ensemble_name, beta = "hermite", 2 # beta = 1, 2, 4
hermite = BetaEnsemble(ensemble_name, beta=beta) # Create the hermite object

hermite_params = {"N":1000} # Size of the matrix
mode = "full" # Full matrix model

hermite.sample(mode, **hermite_params) # Sample

hermite.hist(normalization=True) # Histogram of the eigenvalues/sqrt(beta*N)