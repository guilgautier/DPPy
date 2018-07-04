from beta_ensembles import *

ensemble_name, beta = "circular", 3 # beta \in N^*
circular = BetaEnsemble(ensemble_name, beta=beta) # Create the circular object

# First, plot the eigenvalues
circular_params = {"size":30} # Number of points N
sampling_mode = "banded" # banded (quindiagonal) model

# Plot the eigenvalues for increasing beta to see the cristallization
for b in  (1, 10, 20):

	circular.beta = b
	circular.flush_samples()
	circular.sample(sampling_mode, **circular_params) # Sample
	circular.plot() # Plot of the eigenvalues