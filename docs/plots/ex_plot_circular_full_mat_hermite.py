from beta_ensembles import CircularEnsemble

circular = CircularEnsemble(beta=2) # beta must be in {0,1,2,4}, default beta=2

# 1. Plot the eigenvalues, they lie on the unit circle
circular.sample_full_model(size_N=30, haar_mode='Hermite') # Sample
circular.plot() # Plot of the eigenvalues

# 2. Histogram of the angle of more points, should look uniform on [0,2pi]
circular.flush_samples() # Flush previous sample

circular.sample_full_model(size_N=1000, haar_mode='Hermite') # Sample
circular.hist()