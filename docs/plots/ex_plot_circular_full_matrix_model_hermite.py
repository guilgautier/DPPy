from dppy.beta_ensembles.beta_ensembles import CircularBetaEnsemble

circular = CircularBetaEnsemble(beta=2)  # beta in {0,1,2,4}, default beta=2

# 1. Plot the eigenvalues, they lie on the unit circle
circular.sample_full_model(size_N=30, haar_mode="Hermite")
circular.plot()

# 2. Histogram of the angle of more points, should look uniform on [0,2pi]
circular.flush_samples()  # Flush previous sample

circular.sample_full_model(size_N=1000, haar_mode="Hermite")
circular.hist()
