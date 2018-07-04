from beta_ensembles import *

ensemble_name, beta = "circular", 2 # beta = 1, 2, 4
circular = BetaEnsemble(ensemble_name, beta=beta) # Create the circular object

# First, plot the eigenvalues
circular_params = {"N":30, "mode":"QR"} #/ "mode":"hermite"
sampling_mode = "full" # Full matrix model

circular.sample(sampling_mode, **circular_params) # Sample
circular.plot() # Plot of the eigenvalues

# Then, display the histogram of more eigenvalues
circular.flush_samples()

circular_params["N"] = 1000
sampling_mode = "full" # Full matrix model

circular.sample(sampling_mode, **circular_params) # Sample
circular.hist() # Histogram of the eigenvalues