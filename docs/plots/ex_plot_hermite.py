from beta_ensembles import *

ensemble_name, beta = "hermite", 2
hermite = BetaEnsemble(ensemble_name, beta=beta)

# Histo
hermite_params = {"N":2000}
sampling_mode = "full"

hermite.sample(sampling_mode, **hermite_params)
hermite.hist(normalization=True)