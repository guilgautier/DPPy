# coding: utf8
"""Implementation of the class :class:`JacobiProjectionDPP` used in :cite:`GaBaVa19` for Monte Carlo with Determinantal Point Processes

- :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.sample` to generate samples
- :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.correlation_kernel` to evaluate the corresponding projection kernel
- :py:meth:`~dppy.continuous.jacobi.JacobiProjectionDPP.plot` to display 1D or 2D samples
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
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

        dim = self.d
        self.a = a if a.size == dim else np.full(dim, a[0], dtype=float)
        self.b = b if b.size == dim else np.full(dim, b[0], dtype=float)

        self._norm_multiD_jacobi = np.prod(
            [
                [norm_jacobi(kn, an, bn) for (kn, an, bn) in zip(k, a, b)]
                for k in self.multi_indices
            ],
            axis=-1,
        )

        self._rejection_bounds = np.prod(
            [
                [bound_jacobi(kn, an, bn) for (kn, an, bn) in zip(k, a, b)]
                for k in self.multi_indices
            ],
            axis=-1,
        )

    def reference_density(self, x):
        # w(x) = (1 - x)^a (1 + x)^b
        x = np.asarray(x)
        a, b = self.a, self.b
        return np.prod((1.0 - x) ** a * (1.0 + x) ** b, axis=-1)

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
        d = self.d
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

        nb_proposals = 3 * 2 ** self.d
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
        dim = self.d

        if dim == 1:
            x = sampler_jacobi_tridiagonal(
                a=a + 1.0,
                b=b + 1.0,
                size=self.N,
                random_state=rng,
            )
            return 1.0 - 2.0 * x

        # if dim >= 2:
        if not (np.all(np.abs(a) <= 0.5) and np.all(np.abs(b) <= 0.5)):
            raise ValueError(
                f"In dimension d={dim}>=2, Jacobi parameters be in [-0.5, 0.5]^d. Given a={a}, b={b}."
            )
        return super().sample(random_state=rng)

    def plot(self, sample, weighted=""):

        if self.d >= 3:
            raise NotImplementedError("Visualizations in d>=3 not implemented")

        tols = 5e-2 * np.ones((self.d, 2))
        # tols = np.zeros((self.d, 2))
        tols[1, 0] = 8e-2

        weights = np.ones(len(sample))

        if weighted == "BH":
            # w_n = 1 / K(x_i, x_i)
            weights = 1.0 / self.correlation_kernel(sample)

        elif weighted == "EZ":
            Phi_X = self.feature_vector(sample)

            idx = np.tile(np.arange(self.N), (self.N, 1))
            idx = idx[~np.eye(idx.shape[0], dtype=bool)].reshape(self.N, -1)

            # w_n = +/- c det A / det B
            #     = +/- c sgn(det A) sgn(det B) exp(logdet A − logdet B)
            sgn_det_A, log_det_A = np.array(np.linalg.slogdet(Phi_X[idx, 1:]))
            sgn_det_B, log_det_B = np.linalg.slogdet(Phi_X)

            np.exp(log_det_A - log_det_B, out=weights)
            weights *= sgn_det_A * sgn_det_B
            weights[1::2] *= -1

        weights /= max(weights.min(), weights.max(), key=abs)

        ticks_pos = [-1, 0, 1]
        ticks_labs = list(map(str, ticks_pos))

        if self.d == 1:

            fig, ax_main = plt.subplots(figsize=(6, 4))

            ax_main.tick_params(axis="both", which="major", labelsize=18)
            ax_main.set_xticks(ticks_pos)
            ax_main.set_xticklabels(ticks_labs)

            ax_main.spines["right"].set_visible(False)
            ax_main.spines["top"].set_visible(False)

            ax_main.scatter(sample[:, 0], np.zeros_like(sample[:, 0]), s=weights)

            ax_main.hist(
                sample[:, 0],
                bins=10,
                weights=weights,
                density=True,
                orientation="vertical",
                alpha=0.5,
            )

            # Top densities
            X_ = np.linspace(-1 + tols[0, 1], 1 - tols[0, 0], 200)[:, None]
            ax_main.plot(
                X_,
                0.5 * stats.beta(1 + self.a, 1 + self.b).pdf(0.5 * (1 - X_)),
                ls="--",
                c="red",
                lw=3,
                alpha=0.7,
                label=r"$a_1 = {:.2f}, b_1 = {:.2f}$".format(self.a, self.b),
            )

            x_lim = ax_main.get_xlim()
            y_lim = ax_main.get_ylim()

            if not weighted:

                tol = 5e-2
                X_ = np.linspace(-1 + tol, 1 - tol, 200)[:, None]
                ax_main.plot(
                    X_,
                    0.5 * stats.beta(0.5, 0.5).pdf(0.5 * (1 - X_)),
                    c="orange",
                    ls="-",
                    lw=3,
                    label=r"$a = b = -0.5$",
                )

            ax_main.legend(
                fontsize=15,
                loc="center",
                bbox_to_anchor=(0.5, -0.15 if weighted else -0.17),
                labelspacing=0.1,
                frameon=False,
            )

        elif self.d == 2:

            # Create Fig and gridspec
            fig = plt.figure(figsize=(6, 6))
            grid = plt.GridSpec(6, 6, hspace=0.0, wspace=0.0)

            ax_main = fig.add_subplot(
                grid[1:, :-1],
                xticks=ticks_pos,
                xticklabels=ticks_labs,
                yticks=ticks_pos,
                yticklabels=ticks_labs,
            )

            ax_main.tick_params(axis="both", which="major", labelsize=18)

            if weighted == "EZ":
                weights *= 100
                w_geq_0 = weights >= 0
                ax_main.scatter(
                    sample[w_geq_0, 0],
                    sample[w_geq_0, 1],
                    s=weights[w_geq_0],
                    alpha=0.7,
                )

                ax_main.scatter(
                    sample[~w_geq_0, 0],
                    sample[~w_geq_0, 1],
                    s=-weights[~w_geq_0],
                    alpha=0.7,
                )
            else:
                weights *= 20
                ax_main.scatter(sample[:, 0], sample[:, 1], s=weights, alpha=0.8)

            x_lim = ax_main.get_xlim()
            y_lim = ax_main.get_ylim()

            # Top plot
            ax_top = fig.add_subplot(
                grid[0, :-1],
                xticks=ticks_pos,
                xticklabels=[],
                yticks=[],
                yticklabels=[],
                frameon=False,
            )
            ax_top.set_xlim(x_lim)

            # Top histogram
            ax_top.hist(
                sample[:, 0],
                bins=10,
                weights=np.abs(weights),
                density=True,
                orientation="vertical",
                alpha=0.5,
            )

            # Top densities
            X_ = np.linspace(-1 + tols[0, 1], 1 - tols[0, 0], 200)[:, None]
            (l_top,) = ax_top.plot(
                X_,
                0.5 * stats.beta(1 + self.a, 1 + self.b).pdf(0.5 * (1 - X_)),
                ls="--",
                c="red",
                lw=3,
                alpha=0.7,
            )

            # Right plot
            ax_right = fig.add_subplot(
                grid[1:, -1],
                xticks=[],
                xticklabels=[],
                yticks=ticks_pos,
                yticklabels=[],
                frameon=False,
            )
            ax_right.set_ylim(y_lim)

            # Right histogram
            ax_right.hist(
                sample[:, 1],
                bins=10,
                weights=np.abs(weights),
                density=True,
                orientation="horizontal",
                alpha=0.5,
            )

            # Right densities
            X_ = np.linspace(-1 + tols[1, 1], 1 - tols[1, 0], 200)[:, None]
            (l_right,) = ax_right.plot(
                0.5 * stats.beta(1 + self.a, 1 + self.b).pdf(0.5 * (1 - X_)),
                X_,
                ls="--",
                c="green",
                lw=3,
                alpha=0.7,
            )

            leg_axes = [l_top, l_right]
            leg_text = [
                ", ".join(
                    [
                        r"$a_{} = {:.2f}$".format(i + 1, a),
                        r"$b_{} = {:.2f}$".format(i + 1, b),
                    ]
                )
                for i, (a, b) in enumerate(zip(self.a, self.b))
            ]

            if not weighted:

                tol = 5e-2
                X_ = np.linspace(-1 + tol, 1 - tol, 200)[:, None]
                (l_arcsine,) = ax_top.plot(
                    X_,
                    0.5 * stats.beta(0.5, 0.5).pdf(0.5 * (1 - X_)),
                    c="orange",
                    ls="-",
                    lw=3,
                )
                ax_right.plot(
                    0.5 * stats.beta(0.5, 0.5).pdf(0.5 * (1 - X_)),
                    X_,
                    c="orange",
                    ls="-",
                    lw=3,
                )

                leg_axes.append(l_arcsine)
                leg_text.append(r"$a = b = -0.5$")

            ax_main.legend(
                leg_axes,
                leg_text,
                fontsize=15,
                loc="center",
                bbox_to_anchor=(0.5, -0.15 if weighted else -0.18),
                labelspacing=0.1,
                frameon=False,
            )


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
        if n == 0:
            log_bound = 0.0
        else:
            log_bound = np.log(2.0)

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
