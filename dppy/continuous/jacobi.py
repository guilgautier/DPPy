# coding: utf8
"""Implementation of the class :class:`JacobiProjectionDPP` used in :cite:`GaBaVa19` for Monte Carlo with Determinantal Point Processes

- :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.sample` to generate samples
- :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.correlation_kernel` to evaluate the corresponding projection kernel
- :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.plot` to display 1D or 2D samples
"""

import numpy as np
from scipy.special import betaln, eval_jacobi, gammaln

from dppy.beta_ensembles.jacobi import sampler_jacobi_tridiagonal
from dppy.continuous.abstract_continuous_dpp import (
    AbstractSpectralContinuousProjectionDPP,
)
from dppy.utils import check_random_state


class JacobiProjectionDPP(AbstractSpectralContinuousProjectionDPP):
    r"""Class representing the continuous projection DPP with eigenfunctions defined as product of orthonormal Jacobi polynomials :math:`\phi_n^{(i)} = P_n^{a_i, b_i}`.

    - reference density

        .. math::
            w(x)
            = \prod_{i=1}^{d}
                (1 - x_i)^{a_i}
                (1 + x_i)^{b_i}
                1_{[-1, 1]}(x_i),

    - kernel :math:`K` (see also :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.correlation_kernel`)

        .. math::
            K(x, y)
            = \sum_{\mathfrak{b}(k)=0}^{N-1}
                \prod_{i=1}^{d}
                P_{k_i}^{(a_i, b_i)}(x) P_{k_i}^{(a_i, b_i)}(y)

        where

        - :math:`P_{n}^{(a_i, b_i)}(u)` correspond to orthonormal Jacobi polynomials

            .. math::
                \int_{-1}^{1}
                    P_{n}^{(a_i, b_i)}(u) P_{m}^{(a_i, b_i)}(u)
                    (1 - u)^{a_i} (1 + u)^{b_i} d u
                = \delta_{nm}.
    """

    def __init__(self, multi_indices, a, b):
        r"""Initialize the continuous Jacobi projection DPP with eigenfunctions indexed by ``multi_indices`` and defined by Jacobi parameters ``a`` and ``b``.

        :param multi_indices: Matrix of size :math:`N \times d` collecting the multi-indices :math:`k` indexing the eigenfunctions :math:`\phi_k`.
        :type multi_indices: array_like

        :param a: first Jacobi parameters :math:`(a_1, \dots, a_d)`.
        :type a: scalar or array_like

        :param b: first Jacobi parameters :math:`(b_1, \dots, b_d)`.
        :type b: scalar or array_like
        """
        super().__init__(multi_indices, dtype_kernel=float)

        a = np.ravel(a)
        b = np.ravel(b)

        if not (np.all(a > -1) and np.all(b > -1)):
            raise ValueError(f"Jacobi parameters must be > -1. Given a={a}, b={b}.")

        dim = self.dimension
        self.a = a if a.size == dim else np.full(dim, a[0], dtype=float)
        self.b = b if b.size == dim else np.full(dim, b[0], dtype=float)

        self._norm_multiD_jacobi = _norm_multiD_jacobi(
            self.multi_indices, self.a, self.b
        )

        self._rejection_bounds = _rejection_bounds_jacobi(
            self.multi_indices, self.a, self.b
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(multi_indices={self.multi_indices.tolist()}, a={self.a.tolist()}, b={self.b.tolist()})"

    def new_multi_indices(self, multi_indices):
        return self.__class__(multi_indices, self.a, self.b)

    def reference_density(self, x):
        # w(x) = (1 - x)^a (1 + x)^b 1_[-1, 1]
        a, b = self.a, self.b
        y = np.clip(x, -1.0, 1.0)
        return np.prod((1.0 - y) ** a * (1.0 + y) ** b, axis=-1)

    def eigen_function_1D(self, n, dim, x):
        a, b = self.a[dim], self.b[dim]
        Pn_x = eval_jacobi(n, a, b, x)
        Pn_x /= norm_jacobi(n, a, b)
        return Pn_x

    def eigen_function(self, idx, x):
        a, b = self.a, self.b
        k = self.multi_indices[idx]
        Pk_x = eval_jacobi(k, a, b, x)
        Pk_norm = self._norm_multiD_jacobi[idx]
        Pk_x /= Pk_norm
        return np.prod(Pk_x)

    def feature_vector(self, x):
        a, b = self.a, self.b
        k = self.multi_indices
        Pk_x = np.prod(eval_jacobi(k, a, b, x), axis=-1)
        Pk_norm = self._norm_multiD_jacobi
        Pk_x /= Pk_norm
        return Pk_x

    def _sample_marginal_proposal(self, random_state=None):
        r"""Generate a sample with distribution

        .. math::
            w_{eq}(x) d x
                = \prod_{i=1}^{d}
                    \frac{1}{\pi\sqrt{1-(x_i)^2}}
                    1_{[-1, 1]}(x_i)
                    d x_i

        to be used as a proposal in :py:meth:`~dppy.continuous.abstract_continuous_dpp.AbstractSpectralContinuousProjectionDPP.sample_marginal`.

        :return: sampled point
        :rtype: array_like
        """
        # Propose x ~ w_eq(x) = \prod_{i=1}^{d} 1/pi 1/sqrt(1-(x_i)^2)
        rng = check_random_state(random_state)
        d = self.dimension
        # rng.beta defined as beta(a, b) propto x^(a-1) (1-x)^(b-1)
        sample = 1.0 - 2.0 * rng.beta(0.5, 0.5, size=d)
        likelihood = np.prod(np.sqrt(1.0 - sample ** 2))
        likelihood /= np.pi ** d
        return sample, likelihood

    def sample_marginal(self, random_state=None):
        # Rejection sampling
        # target: 1/N K(x, x) w(x)
        # proposal: weq(x) = 1 / (pi sqrt(1 - x^2))
        # rejection bound <= 2^d / N
        rng = check_random_state(random_state)

        # Select mode of marginal distribution
        # 1/N K(x, x) w(x) = w(x) sum_k 1/N (P_k(x) / ||P_k||)^2
        idx = rng.randint(self.N)  # k = self.multi_indices[idx]
        Pk = lambda x: self.eigen_function(idx, x)  # already normed

        # target density w(x) (P_k(x) / ||P_k||)^2
        w = self.reference_density
        target = lambda x: w(x) * Pk(x) ** 2

        # Compute upper bound w(x) (P_k(x) / ||P_k||)^2 / weq(x)
        proposal = self._sample_marginal_proposal
        rej_bound = self._rejection_bounds[idx]

        nb_proposals = 3 * 2 ** self.dimension
        for it in range(nb_proposals):
            x, proposal_x = proposal(random_state=rng)
            target_x = target(x)
            accept = rng.rand() * rej_bound < target_x / proposal_x
            if accept or it == nb_proposals - 1:
                return x
        else:
            print(
                f"Failed to sample from the marginal distribution 1/N K(x,x) w(x), after {nb_proposals} proposals. Last proposed point x is returned."
            )

    def sample(self, random_state=None):
        rng = check_random_state(random_state)
        a, b = self.a, self.b
        dim = self.dimension

        if dim == 1:
            x = sampler_jacobi_tridiagonal(
                beta=2,
                size=self.N,
                a=a + 1.0,
                b=b + 1.0,
                random_state=rng,
            )
            return 1.0 - 2.0 * x

        # if dim >= 2:
        if not (np.all(np.abs(a) <= 0.5) and np.all(np.abs(b) <= 0.5)):
            raise ValueError(
                f"In dimension d={dim}>=2, Jacobi parameters be in [-0.5, 0.5]^d. Given a={a}, b={b}."
            )
        return super().sample(random_state=rng)


def norm_jacobi(n, a, b, log_scale=False):
    """Return the norm of the Jacobi polynomial :math:`\| p_n{(a, b)} \|^2`.

    :param n: degree of the Jacobi polynomial.
    :type n: int

    :param a: first parameter of the Jacobi polynomial.
    :type a: float

    :param b: secong parameter of the Jacobi polynomial.
    :type b: float

    :return: :math:`\| p_n{(a, b)} \|^2`.
    :rtype: float
    """
    log_sq_norm = -np.inf

    arcsine = a == b == -0.5
    if arcsine:
        # |P_0|^2 = pi
        if n == 0:
            log_sq_norm = np.log(np.pi)
        else:
            # |P_n|^2 = 0.5 * (gamma(n + 0.5) / factorial(n))**2
            log_sq_norm = 2.0 * (gammaln(n + 0.5) - gammaln(n + 1)) - np.log(2)

    else:
        # |P_n|^2 =
        #   2^(a + b + 1)      Gamma(n + 1 + a) Gamma(n + 1 + b)
        #   n! (2n + a + b + 1)     Gamma(n + 1 + a + b)
        if n == 0:
            log_sq_norm = (a + b + 1) * np.log(2) + betaln(a + 1, b + 1)
        else:
            log_sq_norm = (
                (a + b + 1) * np.log(2)
                + gammaln(n + 1 + a)
                + gammaln(n + 1 + b)
                - gammaln(n + 1)
                - np.log(2 * n + 1 + a + b)
                - gammaln(n + 1 + a + b)
            )
    log_norm = 0.5 * log_sq_norm
    return log_norm if log_scale else np.exp(log_norm)


def bound_jacobi(n, a, b, log_scale=False):
    # Bound on pi (1 - x)^(a + 1/2) (1 + x)^(b + 1/2) (P_n / ||P_n||)^2

    log_bound = -np.inf

    arcsine = a == b == -0.5
    if arcsine:
        log_bound = 0.0 if n == 0 else np.log(2.0)

    else:
        mode = (b - a) / (a + b + 1)
        if n == 0:
            log_norm_P0 = norm_jacobi(n, a, b, log_scale=True)
            log_bound = (
                np.log(np.pi)
                + (0.5 + a) * np.log(1 - mode)
                + (0.5 + b) * np.log(1 + mode)
                - 2.0 * log_norm_P0
            )
        else:
            #   2 * Gamma(n + 1 + a + b) Gamma(n + 1 + max(a,b))
            #   n! * (n+(a+b+1)/2)^(2 * max(a,b)) * Gamma(n + 1 + min(a,b))
            min_a_b = min(a, b)
            max_a_b = max(a, b)

            log_bound = (
                np.log(2)
                + gammaln(n + 1 + a + b)
                + gammaln(n + 1 + max_a_b)
                - gammaln(n + 1)
                - 2 * max_a_b * np.log(n + 0.5 * (a + b + 1))
                - gammaln(n + 1 + min_a_b)
            )

    return log_bound if log_scale else np.exp(log_bound)


def _norm_multiD_jacobi(multi_indices, a, b):
    return np.prod(
        [
            [norm_jacobi(kn, an, bn) for (kn, an, bn) in zip(k, a, b)]
            for k in multi_indices
        ],
        axis=-1,
    )


def _rejection_bounds_jacobi(multi_indices, a, b):
    return np.prod(
        [
            [bound_jacobi(kn, an, bn) for (kn, an, bn) in zip(k, a, b)]
            for k in multi_indices
        ],
        axis=-1,
    )
