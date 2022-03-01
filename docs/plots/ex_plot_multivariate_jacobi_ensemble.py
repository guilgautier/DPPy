import matplotlib.pyplot as plt
import numpy as np

from dppy.continuous.jacobi import JacobiProjectionDPP

# The .plot() method outputs smtg only in dimension d=1 or 2

# Number of points / dimension
N, d = 200, 2
# Jacobi parameters in [-0.5, 0.5]^{d x 2}
jac_params = np.array([[0.5, 0.5], [-0.3, 0.4]])

dpp = JacobiProjectionDPP(N, jac_params)

# Get an exact sample
sampl = dpp.sample()

# Display
# the cloud of points
# the base probability densities
# the marginal empirical histograms
dpp.plot(sample=sampl, weighted=False)
plt.tight_layout()

dpp.plot(sample=sampl, weighted="BH")
plt.tight_layout()

dpp.plot(sample=sampl, weighted="EZ")
plt.tight_layout()
