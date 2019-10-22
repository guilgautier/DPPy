import numpy as np
import matplotlib.pyplot as plt
from dppy.multivariate_jacobi_ope import MultivariateJacobiOPE

# The .plot() method outputs smtg only in dimension d=1 or 2

# Number of points / dimension
N, d = 200, 2
# Jacobi parameters in [-0.5, 0.5]^{d x 2}
jac_params = np.array([[0.5, 0.5],
                       [-0.3, 0.4]])

dpp = MultivariateJacobiOPE(N, jac_params)

# Get an exact sample
sampl = dpp.sample()

# Display
# the cloud of points
# the base probability densities
# the marginal empirical histograms
dpp.plot(sample=sampl, weighted=False)
plt.tight_layout()

# Attach a weight 1/K(x,x) to each of the points
dpp.plot(sample=sampl, weighted=True)
plt.tight_layout()
