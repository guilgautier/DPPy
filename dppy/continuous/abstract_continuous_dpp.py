from abc import ABCMeta, abstractmethod

import numpy as np

from dppy.utils import check_random_state


class AbstractSpectralContinuousProjectionDPP(metaclass=ABCMeta):
    def __init__(self, multi_indices, eltype_kernel=float):
        multi_indices = np.array(multi_indices)
        assert multi_indices.ndim == 2 and multi_indices.dtype == int
        self.multi_indices = multi_indices
        self.N, self.d = self.multi_indices.shape
        self.eltype_kernel = eltype_kernel

    @abstractmethod
    def eigen_function_1d(self, i, x):
        # i int
        # x float or complex
        raise NotImplementedError()

    def eigen_function_multi(self, multi_idx, x):
        # multi_idx (d, )
        # x (d, )
        phi = self.eigen_function_1d
        return np.prod([phi(i, xi) for i, xi in zip(multi_idx, x)])

    def eigen_function(self, n, x):
        # n int
        # x (d, )
        multi_n = self.multi_indices[n]
        return self.eigen_function_multi(multi_n, x)

    def feature_vector(self, x):
        # x (d, )
        phi = self.eigen_function
        return np.array([phi(n, x) for n in range(self.N)])

    def correlation_kernel(self, x, y):
        # x (d, ) or (n, d)
        # y (d, ) or (n, d)
        X, Y = np.atleast_2d(x, y)

        n, dx = X.shape
        m, dy = Y.shape
        assert dx == dy == self.d

        phi = self.feature_vector
        K = np.zeros((n, m), dtype=self.eltype_kernel)
        for i, xi in enumerate(X):
            for j, yj in enumerate(Y):
                K[i, j] = np.vdot(phi(yj), phi(xi))

        # phi = self.feature_vector
        # phi_x = phi(x)
        # phi_y = phi_x if y is None else phi(y)
        # return np.dot(phi_x, phi_y.conj().T)

        return K.squeeze()

    @abstractmethod
    def sample_marginal(self, random_state=None):
        # rng = check_random_state(random_state)
        raise NotImplementedError()

    def sample(self, random_state=None, nb_trials_max=10_000):
        rng = check_random_state(random_state)

        N, d = self.N, self.d
        sample = np.zeros((N, d))
        phi = np.zeros((N, N), self.eltype_kernel)

        for n in range(N):
            for trial in range(nb_trials_max):
                sample[n] = self.sample_marginal(random_state=rng)

                phi[n] = self.feature_vector(sample[n])
                K_xx = np.vdot(phi[n], phi[n]).real
                if n > 0:
                    Phi = phi[:n]
                    phi[n] -= Phi.T.dot(np.dot(Phi.conj(), phi[n]))
                schur = np.vdot(phi[n], phi[n]).real

                if rng.rand() < schur / K_xx:
                    phi[n] /= np.sqrt(schur)
                    break
            else:
                print(
                    f"conditional x_{n+1} | x_1,...,x_{n}, rejection fails after {trial} proposals"
                )

        return sample
