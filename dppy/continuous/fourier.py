import numpy as np

from dppy.continuous.abstract_continuous_dpp import (
    AbstractSpectralContinuousProjectionDPP,
)
from dppy.utils import check_random_state


class FourierProjectionDPP(AbstractSpectralContinuousProjectionDPP):
    r"""Class representing the continuous projection DPP associated with Fourier eigenfunctions :math:`\phi_k(x) = \exp(2 j \pi \left\langle k, x \right\rangle)`.

    - reference density :math:`w(x) = \prod_{i=1}^{d} 1_{[-1/2, 1/2]}(x_i)`.
    - projection kernel

        .. math::
            K(x, y)
            = \sum_{\mathfrak{b}(k)=0}^{N-1}
                \exp(2 j \pi \left\langle k, x - y \right\rangle).

    .. seealso::

        - :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP` for more details.
    """

    def __init__(self, multi_indices):
        r"""Initialize the continuous Fourier projection DPP with eigenfunctions indexed by ``multi_indices``.

        :param multi_indices: Matrix of size :math:`N \times d` collecting the multi-indices :math:`k` indexing the eigenfunctions :math:`\phi_k`.
        :type multi_indices: array_like
        """
        super().__init__(multi_indices, dtype_kernel=complex)

    def __repr__(self):
        return f"{self.__class__.__name__}(multi_indices={self.multi_indices.tolist()})"

    def new_multi_indices(self, multi_indices):
        return self.__class__(multi_indices)

    def reference_density(self, x):
        return np.ones_like(x, dtype=float)

    def eigen_function_1D(self, n, dim, x):
        # phi_n(x) = exp(2 j pi n x)
        # same eigen function family across dimension
        raise np.exp(2 * np.pi * 1j * n * x)

    def eigen_function_multiD(self, multi_idx, x):
        # phi_k(x) = exp(2 j pi <k, x>)
        X, K = np.atleast_2d(x, multi_idx)
        phi = np.zeros((X.shape[0], K.shape[0]), dtype=complex)
        phi.imag = np.dot(X, K.T)
        phi *= 2 * np.pi
        # np.conj(K_X, out=K_X)
        np.exp(phi, out=phi)
        return phi.squeeze()

    def feature_vector(self, x):
        return self.eigen_function_multiD(self.multi_indices, x)

    def correlation_kernel(self, x, y=None):
        if (y is None or y is x) and x.shape == (self.dimension,):
            return float(self.N)

        phi = self.feature_vector
        phi_x = phi(x)
        phi_y = phi_x if y is None else phi(y)
        return np.dot(phi_x, phi_y.conj().T)

    def sample_marginal(self, random_state=None):
        rng = check_random_state(random_state)
        return rng.rand(self.dimension)
