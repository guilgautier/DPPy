from dppy.beta_ensembles.jacobi import HermiteBetaEnsemble

jacobi = JacobiBetaEnsemble(beta=2)  # beta must be in {0,1,2,4}, default beta=2
jacobi.sample_full_model(N=400, M1=500, M2=600)  # M_1, M_2 >= N
# jacobi.plot(normalization=True)
jacobi.hist(normalization=True)

# To compare with the sampling speed of the triadiagonal model simply use
# jacobi.sample_banded_model(N=400, M1=500, M2=600)
