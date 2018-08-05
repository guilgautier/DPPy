from beta_ensembles import *

ensemble_name, beta = "circular", 2 # beta = 1, 2, 4
circular = BetaEnsemble(ensemble_name, beta=beta) # Create the circular object

# First, plot the eigenvalues
circular_params = {"N":30, "haar_mode":"hermite"} #/ "haar_mode":"QR"
mode = "full" # Full matrix model

circular.sample(mode, **circular_params) # Sample
circular.plot() # Plot of the eigenvalues

# Then, display the histogram of more eigenvalues
circular.flush_samples()

circular_params["N"] = 1000
mode = "full" # Full matrix model

circular.sample(mode, **circular_params) # Sample
circular.hist() # Histogram of the eigenvalues