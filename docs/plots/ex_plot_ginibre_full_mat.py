from beta_ensembles import *

ensemble_name, beta = "ginibre", 2 # beta = 1, 2, 4
ginibre = BetaEnsemble(ensemble_name, beta=beta) # Create the ginibre object

ginibre_params = {"N":100, "mode":"full"} # Matrix size / Full matrix model

ginibre.sample(**ginibre_params) # Sample

ginibre.plot(normalization=True) # Histogram of the eigenvalues/sqrt(N)