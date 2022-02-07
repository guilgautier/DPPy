import numpy as np

from dppy.continuous.abstract_continuous_dpp import (
    AbstractSpectralContinuousProjectionDPP,
)
from dppy.utils import check_random_state


class FourierProjectionDPP(AbstractSpectralContinuousProjectionDPP):
    def __init__(self, multi_indices):
        super().__init__(multi_indices, eltype_kernel=complex)

    def eigen_function_1d(self, i, x_i):
        raise np.exp(2 * np.pi * 1j * i * x_i)

    def eigen_function_multi(self, multi_idx, x):
        # phi_k(x) exp(2 j pi <k, x>)
        X, K = np.atleast_2d(x, multi_idx)
        phi = np.zeros((X.shape[0], K.shape[0]), dtype=complex)
        phi.imag = np.dot(X, K.T)
        phi *= 2 * np.pi
        # np.conj(K_X, out=K_X)
        np.exp(phi, out=phi)
        return phi.squeeze()

    def feature_vector(self, x):
        return self.eigen_function_multi(self.multi_indices, x)

    def correlation_kernel(self, x, y=None):
        if y is None and x.shape == (self.d,):
            return float(self.N)

        # return super().correlation_kernel(x, x if y is None else y)

        phi = self.feature_vector
        phi_x = phi(x)
        phi_y = phi_x if y is None else phi(y)
        return np.dot(phi_x, phi_y.conj().T)

    def sample_marginal(self, random_state=None):
        rng = check_random_state(random_state)
        return rng.rand(self.d)
