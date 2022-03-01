import numpy as np

from dppy.utils import check_random_state


class AbstractSpectralContinuousProjectionDPP:
    def __init__(self, multi_indices, dtype_kernel=float):
        multi_indices = np.atleast_2d(multi_indices)
        assert multi_indices.ndim == 2 and multi_indices.dtype == int

        self.multi_indices = multi_indices
        self.N, self.d = self.multi_indices.shape
        self.dtype_kernel = dtype_kernel

    def eigen_function_1D(self, i, x):
        # i int
        # x float or complex
        raise NotImplementedError()

    def eigen_function_multiD(self, multi_idx, x):
        # multi_idx (d, )
        # x (d, )
        phi = self.eigen_function_1D
        return np.prod([phi(i, xi) for i, xi in zip(multi_idx, x)])

    def eigen_function(self, idx, x):
        # n int
        # x (d, )
        multi_idx = self.multi_indices[idx]
        return self.eigen_function_multiD(multi_idx, x)

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
        K = np.zeros((n, m), dtype=self.dtype_kernel)
        for i, xi in enumerate(X):
            for j, yj in enumerate(Y):
                K[i, j] = np.vdot(phi(yj), phi(xi))

        # phi = self.feature_vector
        # phi_x = phi(x)
        # phi_y = phi_x if y is None else phi(y)
        # return np.dot(phi_x, phi_y.conj().T)

        return K.squeeze()

    def sample_marginal(self, random_state=None):
        # rng = check_random_state(random_state)
        raise NotImplementedError()

    def sample(self, random_state=None, nb_proposals=10_000):
        rng = check_random_state(random_state)

        N, d = self.N, self.d
        sample = np.zeros((N, d))
        phi = np.zeros((N, N), dtype=self.dtype_kernel)

        for n in range(N):
            # Rejection sampling
            # target: conditional w(x) [K(x,x) - K_Yx^* K_Y^-1 K_Yx] / (N-(n-1))
            # proposal: marginal w(x) K(x,x) / N
            # rejection bound: N / (N-(n-1)), simplifies in target / proposal
            for _ in range(nb_proposals):
                sample[n] = self.sample_marginal(random_state=rng)
                phi[n] = self.feature_vector(sample[n])
                proposal_x = np.vdot(phi[n], phi[n]).real

                if n > 0:
                    Phi = phi[:n]
                    phi[n] -= Phi.T.dot(np.dot(Phi.conj(), phi[n]))

                target_x = np.vdot(phi[n], phi[n]).real
                if rng.rand() < target_x / proposal_x:
                    phi[n] /= np.sqrt(target_x)
                    break
            else:
                print(
                    f"Rejection sampling from conditional x_{n+1} | x_1,...,x_{n} failed after {nb_proposals} proposals. The last proposed point was accepted."
                )

        return sample
