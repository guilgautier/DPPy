import numpy as np

from dppy.utils import check_random_state


class AbstractSpectralContinuousProjectionDPP:
    r"""Class representing an abstraction of a continuous projection DPP in dimension :math:`d`, defined by :math:`d` families :math:`(\phi_n^{(1)}(x))_{n \geq 0}, \ldots, :math:`(\phi_n^{(d)}(x))_{n \geq 0}` of orthonormal functions w.r.t. :math:`w^{(1)}(u) d u, \ldots, w^{(d)}(u) d u`.

    - a product reference measure :math:`\mu(dx) = w(x) d x = \prod_{i=1}^{d} w^{(i)}(x_i) d x_i` (see also :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.reference_density`),

    - a projection kernel :math:`K` (see also :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.correlation_kernel`)

        .. math::
            K(x, y)
            = \sum_{\mathfrak{b}(k)=0}^{N-1}
                \phi_{k}(x) \overline{\phi_{k}(y)}
            = \Phi(x)^{*} \Phi(y),

        where

        - :math:`\mathfrak{b}` defines an ordering of the multi-index :math:`k \in \mathbb{N}^d` materialized by the :py:attr:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.multi_indices` of size ``N``,

        - :math:`\phi_{k}(x)` characterizes the eigenfunction associated to the multi-index :math:`k` (see also :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.eigen_function_multiD`). It is defined as the product of 1-dimensional orthonormal functions :math:`\phi_{k}(x) = \prod_{i=1}^d \phi_{k_i}(x_i)`, such that so that :math:`(\phi_{k_i})` are orthonormal w.r.t :math:`w^{(i)}(u)`. In particular :math:`\int P_{k}(x) P_{\ell}(x) w(x) d x = \delta_{k\ell}` (see also :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.eigen_function_1D`).

        - the feature vector :math:`\Phi(x) = \left(P_{\mathfrak{b}^{-1}(0)}(x), \dots, P_{\mathfrak{b}^{-1}(N-1)}(x) \right)^{\top}`.

    .. important::

        Child classes have a basic exact sampling method :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.sample`, provided the methods :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.eigen_function_1D` and :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.sample_marginal` are implemented.
    """

    def __init__(self, multi_indices, dtype_kernel):
        """_summary_

        :param multi_indices:
        :type multi_indices: array_like
        :param dtype_kernel: Type of the kernel entries.
        :type dtype_kernel: type
        """
        multi_indices = np.atleast_2d(multi_indices)
        assert multi_indices.ndim == 2 and multi_indices.dtype == int

        self.multi_indices = multi_indices
        self.dtype_kernel = dtype_kernel

    def __len__(self):
        return self.multi_indices.shape[0]

    @property
    def N(self):
        return len(self)

    @property
    def dimension(self):
        return self.multi_indices.shape[1]

    def reference_density(self, x):
        """Evaluate the reference density :math:`w(x)`.

        :param x: Vector of size :math:`d` where :math:`w` is evaluated
        :type x: array_like
        """
        raise NotImplementedError()

    def eigen_function_1D(self, n, dim, x):
        r"""Evaluate the ``n``-th 1-dimensional eigenfunction  :math:`\phi_{n}^{(i)}(x)` along the :math:`i`-th dimension ``dim`` at point ``x``.

        :param n: Index :math:`n` of the eigenfunction along the :math:`i`-th dimension ``dim``.
        :type n: int
        :param dim: Dimension index in :math:`\{1, \ldots, d\}`.
        :type dim: int
        :param x: Point
        :type x: scalar
        """
        raise NotImplementedError()

    def eigen_function_multiD(self, multi_idx, x):
        r"""Evaluate the eigenfunction :math:`\phi_{k}(x) = \prod_{i=1}^{d} \phi_{k_i}^{(i)}(x_i)` where :math:`k=` ``multi_idx``.

        :param multi_idx: Integer vector of size :math:`d` representing the multi-index :math:`k \in \mathbb{N}^d`.
        :type multi_idx: array_like

        :param x: Vector of size :math:`d`.
        :type x: array_like
        """
        phi = self.eigen_function_1D
        return np.prod(
            [phi(ni, di, xi) for di, (ni, xi) in enumerate(zip(multi_idx, x))]
        )

    def eigen_function(self, idx, x):
        r"""Evaluate the eigenfunction :math:`\phi_{k}(x) = \prod_{i=1}^{d} \phi_{k_i}^{(i)}(x_i)` where :math:`k=\mathfrak{b}` ``(idx)``.

        The multi-index :math:`k` is extracted from the attribute :py:attr:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.multi_indices` ``[idx]``.

        :param idx: Index refering to the multi-index :math:`k=` :py:attr:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.multi_indices` ``[idx]``, in the ordering :math:`\mathfrak{b}`.
        :type idx: int

        :param x: Vector of size :math:`d`.
        :type x: array_like
        """
        multi_idx = self.multi_indices[idx]
        return self.eigen_function_multiD(multi_idx, x)

    def feature_vector(self, x):
        r"""Evaluate the feature vector :math:`\Phi(x) = \left(P_{\mathfrak{b}^{-1}(0)}(x), \dots, P_{\mathfrak{b}^{-1}(N-1)}(x) \right)^{\top}` such that :math:`K(x, y) = \Phi(x)^{*} \Phi(y)`.

        :param x: Point
        :type x: array_like

        :return: Array of size :math:`N` representing the feature vector :math:`\Phi(x)`.
        :rtype: array_like

        .. seealso::

            - :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.eigen_function`
        """
        phi = self.eigen_function
        return np.array([phi(n, x) for n in range(self.N)])

    def correlation_kernel(self, X, Y):
        r"""Evalute the correlation kernel :math:`\left(K(x, y)\right)_{x\in X, y\in Y}` of the projection DPP,

        .. math::
            K(x, y)
            = \sum_{\mathfrak{b}(k)=0}^{N-1}
                \phi_{k}(x) \phi_{k}(y)
            = \Phi(x)^{*} \Phi(y),

        where

        - :math:`k \in \mathbb{N}^d` is a multi-index ordered according to the ordering :math:`\mathfrak{b}` defined by :py:attr:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.multi_indices`.

        - :math:`\Phi(x) = \left(P_{\mathfrak{b}^{-1}(0)}(x), \dots, P_{\mathfrak{b}^{-1}(N-1)}(x) \right)`, see :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.feature_vector`.

        :param X: Points
        :type X: array_like

        :param Y: Points
        :type Y: array_like

        :return: The evaluation of the correlation kernel :math:`\left(K(x, y)\right)_{x\in X, y\in Y}`, defined by.
        :rtype:
            array_like
        """
        # x (d, ) or (n, d)
        # y (d, ) or (n, d)
        X, Y = np.atleast_2d(X, Y)

        n, dx = X.shape
        m, dy = Y.shape
        assert dx == dy == self.dimension

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
        r"""Generate an exact sample from the marginal distribution

        .. math::
            \frac{1}{N} K(x, x) w(x) dx
            = \frac{1}{N}
                \sum_{\mathfrak{b}(k)=0}^{N-1}
                \left( \frac{\Phi_k(x)}{\left\| \Phi_k \right\|} \right)^2
                w(x)

        to be used as a proposal in a the rejection sampling mechanism involved in the main exact sampling method :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.sample`.

        :return: Exact sample vector of size :math:`d`.
        :rtype: array_like
        """
        # rng = check_random_state(random_state)
        raise NotImplementedError()

    def sample(self, random_state=None, nb_proposals=10_000):
        r"""Generate an exact sample from the projection DPP

        Use the chain rule :cite:`HKPV06` (Algorithm 18) to sample :math:`\left(x_{1}, \dots, x_{N} \right)` with density

        .. math::

            & \frac{1}{N!}
                \left(K(x_n,x_p)\right)_{n,p=1}^{N}
                \prod_{n=1}^{N} w(x_n)\\
            &= \frac{1}{N} K(x_1,x_1) w(x_1)
            \prod_{n=2}^{N}
                \frac{
                    K(x_n,x_n)
                    - K(x_n,x_{1:n-1})
                    \left[\left(K(x_k,x_l)\right)_{k,l=1}^{n-1}\right]^{-1}
                    K(x_{1:n-1},x_n)
                    }{N-(n-1)}
                    w(x_n)\\
            &= \frac{\| \Phi(x) \|^2}{N} \omega(x_1) d x_1
            \prod_{n=2}^{N}
                \frac{\operatorname{distance}^2(\Phi(x_n), \operatorname{span}\{\Phi(x_p)\}_{p=1}^{n-1})}
                {N-(n-1)}
            \omega(x_n) d x_n

        The order in which the points were sampled can be forgotten to obtain a valid sample of the corresponding DPP

        - :math:`x_1 \sim \frac{1}{N} K(x,x) w(x)` using :py:meth:`sample_chain_rule_proposal`

        - :math:`x_n | Y = \left\{ x_{1}, \dots, x_{n-1} \right\}`, is sampled using rejection sampling with proposal density :math:`\frac{1}{N} K(x,x) w(x)` and rejection bound \frac{N}{N-(n-1)}

            .. math::

                \frac{1}{N-(n-1)} [K(x,x) - K(x, Y) K_Y^{-1} K(Y, x)] w(x)
                \leq \frac{N}{N-(n-1)} \frac{1}{N} K(x,x) w(x)

        .. note::

            Using the gram structure :math:`K(x, y) = \Phi(x)^{*} \Phi(y)` the numerator of the successive conditionals reads

            .. math::

                K(x, x) - K(x, Y) K(Y, Y)^{-1} K(Y, x)
                &= \operatorname{distance}^2(\Phi(x_n), \operatorname{span}\{\Phi(x_p)\}_{p=1}^{n-1})\\
                &= \left\| (I - \Pi_{\operatorname{span}\{\Phi(x_p)\}_{p=1}^{n-1}} \phi(x)\right\|^2

            which can be computed simply in a vectorized way.
            The overall procedure is akin to a sequential Gram-Schmidt orthogonalization of :math:`\Phi(x_{1}), \dots, \Phi(x_{N})`.

        .. seealso::

            - :ref:`continuous_dpps_exact_sampling_projection_dpp_chain_rule`
            - :py:meth:`sample_chain_rule_proposal`
        """
        rng = check_random_state(random_state)

        N, d = self.N, self.dimension
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
