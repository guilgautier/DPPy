import numpy as np
import itertools as itt

from scipy.special import beta, betaln, factorial, gamma, gammaln
from scipy.special import eval_jacobi
from scipy.special import logsumexp

from dppy.random_matrices import mu_ref_beta_sampler_tridiag\
                                 as tridiagonal_model

# import warnings


class MultivariateJacobiOPE:
    """
    Multivariate Jacobi Orthogonal Polynomial Ensemble used by :cite:`BaHa16` for Monte Carlo with Determinantal Point Processes.

    Reference weight
    .. math::

        w(x) = \prod_{i=1}^{d} (1-x)^{a_i} (1+x)^{b_i}

    :param N:
        Number of points :math:`N \geq 2`
    :type N:
        int

    :param jacobi_params:
        Jacobi parameters :math:`\in [-\frac{1}{2}, \frac{1}{2}]^{d \times 2}`.
        The number of rows :math:`d` prescribes the ambient dimension of the points i.e. :math:`x_{1}, \dots, x_{N} \in [-1, 1]^d`
    :type N:
        array_like

    .. seealso::

        - :cite:`Kon05`
        - :cite:`BaHa16`
    """

    def __init__(self, N, jacobi_params, log_scale=True):

        self.N, self.jacobi_params, self.dim =\
            self._check_params(N, jacobi_params)

        self.ordering = compute_ordering(self.N, self.dim)

        self.deg_max, self.poly_degrees =\
            poly_degrees(np.max(self.ordering, axis=0))

        self.square_norms =\
            compute_square_norms(self.jacobi_params, self.deg_max)

        self.rejection_bound =\
            compute_rejection_bound(self.jacobi_params,
                                    self.ordering,
                                    log_scale)

    def _check_params(self, N, jacobi_params):
        """ Check that:

        - The number of points :math:`N \geq 2`
        - Jacobi parameters :math:`(a_i, b_i)_{i=1}^d \in [-0.5, 0.5]^d`.
        """
        if type(N) is not int or N < 1:
            return TypeError('Number of points N={} is not an integer or < 2'.format(N))

        dim = jacobi_params.size // 2
        if dim == 1:
            war = 'In dimension {}, the tridiagonal model is used'.format(dim)
            # warnings.warn(war)

        if (-0.5 <= jacobi_params).all() and (jacobi_params <= 0.5).all():
            return N, jacobi_params, dim
        else:
            raise ValueError('Jacobi parameters not in [-0.5, 0.5]^d, we have no guaranty')

    def K(self, X, Y=None):
        ''' Compute the orthogonal projection kernel :math:`K` onto the span of the first N polynomials  the

        .. math::
            K(x, y) = \sum_{\alpha}
                        \frac{P_{\alpha}(x)P_{\alpha}(y)}
                             {\|P_{\alpha}\|^2}

        for :math:`P_{\alpha}(x) = \prod_{i=1}^d P_{\alpha_i}^{a_i, b_i}(x_i)`
        '''
        if Y is None:

            if X.size // self.dim == 1:  # X is vector in R^d
                polys_X_2 = eval_jacobi(self.poly_degrees,
                                        self.jacobi_params[:, 0],
                                        self.jacobi_params[:, 1],
                                        X)**2\
                            / self.square_norms

                return np.sum(
                            np.prod(
                                polys_X_2[self.ordering, range(self.dim)],
                            axis=1),
                        axis=0)

            else:
                polys_X_2 = eval_jacobi(self.poly_degrees,
                                        self.jacobi_params[:, 0],
                                        self.jacobi_params[:, 1],
                                        X[:, None])**2\
                            / self.square_norms

                return np.sum(
                            np.prod(
                                polys_X_2[:, self.ordering, range(self.dim)],
                            axis=2),
                        axis=1)

        else:

            lenX = X.size // self.dim  # X.shape[0] if X.ndim > 1 else 1
            lenY = Y.size // self.dim  # Y.shape[0] if Y.ndim > 1 else 1

            polys_X_Y = eval_jacobi(self.poly_degrees,
                                    self.jacobi_params[:, 0],
                                    self.jacobi_params[:, 1],
                                    np.vstack((X, Y))[:, None])

            if lenX > lenY:

                polys_X_Y[:lenX] *= polys_X_Y[lenX:] / self.square_norms

                return np.sum(
                            np.prod(
                                polys_X_Y[:lenX, self.ordering, range(self.dim)],
                            axis=2),
                        axis=1)

            else:  # if lenX < lenY:

                polys_X_Y[lenX:] *= polys_X_Y[:lenX] / self.square_norms

                return np.sum(
                            np.prod(
                                polys_X_Y[lenX:, self.ordering, range(self.dim)],
                            axis=2),
                        axis=1)

    def sample_from_proposal(self, a=0.5, b=0.5):

        return 2.0 * np.random.beta(a, b, size=self.dim) - 1.0

    def eval_w(self, x):

        return np.prod((1 - x)**(self.jacobi_params[:, 0])\
                        * (1 + x)**(self.jacobi_params[:, 1]), axis=-1)

    def eval_w_over_mu_eq(self, x):

        a_b = 0.5 + self.jacobi_params

        return np.pi**self.dim\
                * np.prod((1 - x)**(a_b[:, 0]) * (1 + x)**(a_b[:, 1]), axis=-1)

    def sample(self, nb_trials_max=10000):
        """ Rejection sampling with proposal :math:`\mu_{\text{eq}}^{\otimes d}` where
        .. math::

            \mu_{\text{eq}}
                = \frac{1}{\pi \sqrt{1-x^2}} 1_{(-1, 1)}(x) d x

        """

        if self.dim == 1:
            sample = tridiagonal_model(a=self.jacobi_params[0, 0] + 1,
                                       b=self.jacobi_params[0, 1] + 1,
                                       beta=2,
                                       size=self.N)
            return 1.0 - 2.0 * sample[:, None]

        sample = np.zeros((self.N, self.dim))
        K_xY = np.zeros(self.N)  # [K(x,y) for y in Y]
        K_1 = np.zeros((self.N, self.N))  # K_YY^-1: inverse of [K(x,y)]_{Y,Y}

        nb_not_enough_trials = 0

        for it in range(self.N):
            it_1 = it + 1

            for _ in range(nb_trials_max):

                # Propose a point from arcsine distriu
                sample[it] = self.sample_from_proposal()

                # K_xY, K_xx = K_xY[:it], K_xY[it]
                K_xY[:it_1] = self.K(sample[it], sample[:it_1])

                # schur = K(x, x) - K(x, Y) K(Y, Y)^-1 K(Y, x)
                schur = K_xY[it] - K_xY[:it].dot(K_1[:it, :it]).dot(K_xY[:it])
                # prod pi (1-x)**(a+1/2) (1+x)**(b+1/2)
                w_over_mu_eq = self.eval_w_over_mu_eq(sample[it])

                U = np.random.rand()
                if U * self.rejection_bound < schur * w_over_mu_eq:
                    break

            else:
                nb_not_enough_trials += 1
                sample[it] = 1e-5 * np.random.randn(self.dim)

                K_xY[:it_1] = self.K(sample[it], sample[:it_1])
                schur = K_xY[it] - K_xY[:it].dot(K_1[:it, :it]).dot(K_xY[:it])

            # Updata K_1 using Woodebury formula, from K_Y^-1 to K_{Y+x}^-1
            if it == 0:
                # K^-1(0, 0) = 1 / K(x, x) = 1 / schur
                K_1[0, 0] = 1.0 / K_xY[it]

            elif it == 1:
                # K_1 = [[K(x, x), -K(x, y)],
                #          [-K(x, y), -K(y, y)]]
                #         / K(x, x) K(y, y) - K(x, y)^2
                K_YY = 1 / K_1[0, 0]  # 1/self.K(sample[0])
                K_1[:2, :2] = np.array([[K_xY[it], -K_xY[:it]],
                                        [-K_xY[:it], K_YY]])
                K_1[:2, :2] /= (K_xY[it] * K_YY - K_xY[:it]**2)

            else:  # The following also works for it=0, 1 ;)
                # K_{Y+x}^-1 =
                # [[K_Y^-1 + (K_Y^-1 K_Yx K_xY K_Y^-1)/schur(x),
                #   -K_Y^-1 K_Yx / schur(x)],
                # [-K_xY K_Y^-1/ schur(x),
                #   1/schur(x)]]
                temp = K_1[:it, :it].dot(K_xY[:it])

                K_1[:it, :it] += np.outer(temp, temp / schur)
                K_1[:it, it] = - temp / schur
                K_1[it, :it] = K_1[:it, it]

                K_1[it, it] = 1.0 / schur

        if nb_not_enough_trials:
            print('not_enough_trials=', nb_not_enough_trials)

        return sample

    def eval_polys(self, X):

        if X.size // self.dim == 1:  # X is vector in R^d
            polys_X = eval_jacobi(self.poly_degrees,
                                    self.jacobi_params[:, 0],
                                    self.jacobi_params[:, 1],
                                    X)\
                        / np.sqrt(self.square_norms)

            return np.prod(polys_X[self.ordering, range(self.dim)],
                           axis=1)

        else:
            polys_X = eval_jacobi(self.poly_degrees,
                                    self.jacobi_params[:, 0],
                                    self.jacobi_params[:, 1],
                                    X[:, None])\
                        / np.sqrt(self.square_norms)

            return np.prod(polys_X[:, self.ordering, range(self.dim)],
                           axis=2)


def compute_ordering(N, d):
    """ :cite:`BaHa16` Section 2.1.3
    """
    layer_max = np.floor(N**(1.0 / d)).astype(np.int16)

    ordering = itt.chain.from_iterable(
                filter(lambda x: m in x, itt.product(range(m + 1), repeat=d))
                for m in range(layer_max + 1))

    return list(ordering)[:N]


def compute_square_norms(jacobi_params, deg_max):
    """ Compute square norms :math:`\|P_{\alpha}\|^2 for each :math:`\alpha \in` `ordering` of multivariate polynomials orthogonal with respect to the weight function :math:`\prod_{i=1}^d (1-x)^{a_i} (1+x)^{b_i}`

    .. math::

        \|P_{\alpha}\|^2
            = \prod_{i=1}^d \| P_{\alpha_i}^{(a_i,b_i)}\|^2

    .. seealso::

        - :ref:` Wikipedia Jacobi polynomials <https://en.wikipedia.org/wiki/Jacobi_polynomials#Orthogonality>`__
    """

    # Initialize
    # - [square_norms]_ij = ||P_i^{a_j, b_j}||^2
    # - [bounds]_ij on
    #       pi (1-x)^(a_j+1/2) (1+x)^(b_j+1/2) P_i^2/||P_i||^2
    dim = jacobi_params.size // 2
    square_norms = np.zeros((deg_max + 1, dim))

    n = np.arange(1, deg_max + 1)[:, None]

    arcsine = np.all(jacobi_params == -0.5, axis=1)
    if any(arcsine):
        # |P_0|^2 = pi
        # |P_n|^2 = 1/2 (Gamma(n+1/2)/n!)^2 otherwise
        square_norms[0, arcsine] = np.pi
        square_norms[1:, arcsine] =\
            0.5 * np.exp(2 * (gammaln(n + 0.5) - gammaln(n + 1)))
        # 0.5 * (gamma(n + 0.5) / factorial(n))**2

    non_arcsine = np.any(jacobi_params != -0.5, axis=1)
    if any(non_arcsine):
        # |P_n|^2 =
        #   2^(a + b + 1)      Gamma(n + 1 + a) Gamma(n + 1 + b)
        #   n! (2n + a + b + 1)     Gamma(n + 1 + a + b)
        a = jacobi_params[non_arcsine, 0]
        b = jacobi_params[non_arcsine, 1]

        square_norms[0, non_arcsine] =\
            2**(a + b + 1) * beta(a + 1, b + 1)

        square_norms[1:, non_arcsine] =\
            np.exp(
                (a + b + 1) * np.log(2)
                + gammaln(n + 1 + a)
                + gammaln(n + 1 + b)
                - gammaln(n + 1)
                - np.log(2 * n + 1 + a + b)
                - gammaln(n + 1 + a + b)
                )

            # 2**(a + b + 1) * gamma(n + 1 + a) * gamma(n + 1 + b)\
            # /(factorial(n) * (2 * n + 1 + a + b)  * gamma(n + 1 + a + b))

    # |P_alpha|^2 = \prod_{i=1}^d |P_{alpha_i}^{a_i,b_i}|^2
    return square_norms


def compute_rejection_bound(jacobi_params, ordering, log_scale=False):
    """ Compute the rejection constant for the rejection sampling scheme with proposal distribution
    :math:`\mu_{eq}^{\otimes d} \prod_{i=1}^{d} w_{eq}(x_i)` where :math:`w_{eq}(x) = \frac{1}{\pi \sqrt{1-x}}`.

    Applying the chain rule, conditionals have the following form
    .. math::

        &\frac{K(x,x) - K(x, Y) K_Y^{-1} K(Y, x)}{N-|Y|} w(x)
            \frac{1}{\frac{w_{\text{eq}}(x)}{\pi^d}}\\
        &\leq \frac{\pi^d}{N-|Y|} \frac{K(x,x) w(x)}{w_{\text{eq}}(x)}
        &= \frac{1}{N-|Y|}
            \sum_{\alpha}
                \prod_{i=1}^{d}
                    \pi
                    \lrp{1 - x_i}^{a_i + \frac12}
                    \lrp{1 + x_i}^{b_i + \frac12}
                    \frac{P_{\alpha_i}^{a_i, b_i}(x_i)^2}
                         {\|P_{\alpha_i}^{a_i, b_i}\|^2}\\

    each term of the product can be bounded using :cite:`Gau09`
    .. math::

        \pi \frac{w^{a,b}(x)}{w_{\text{eq}}(x)}
            \left[\hat{P}_{n}^{(a, b)}(x)\right]^{2}
        &=
            \pi
            (1-x)^{a+\frac{1}{2}}
            (1+x)^{b+\frac{1}{2}}
        \left[\hat{P}_{n}^{(a, b)}(x)\right]^{2}\\
        &\leq
            \frac{2}
                  {n!(n+(a+b+1) / 2)^{2 b}}
            \frac{\Gamma(n+a+b+1)
                    \Gamma(n+b+1)}
                  {\Gamma(n+a+1)},
        \quad|a|,|b| \leq \frac{1}{2}

    .. see also::

        - :cite:`Gau09` for the domination
    """

    # Initialize [bounds]_ij on
    # pi (1-x)^(a_j+1/2) (1+x)^(b_j+1/2) P_i^2/||P_i||^2
    deg_max, dim = np.max(ordering), jacobi_params.size // 2
    bounds = np.zeros((deg_max + 1, dim))

    arcsine = np.all(jacobi_params == -0.5, axis=1)
    if any(arcsine):
        bounds[0, arcsine] = 0.0 if log_scale else 1.0
        bounds[1:, arcsine] = np.log(2.0) if log_scale else 2.0

    non_arcsine = np.any(jacobi_params != -0.5, axis=1)
    if any(non_arcsine):
        # bounds[non_arcsine, 0]
        #   = pi * (1-mode)^(a+1/2) (1+mode)^(b+1/2) * 1 / ||P_0||^2
        # where mode = argmax (1-x)^(a+1/2) (1+x)^(b+1/2) = (b-a)/(a+b+1)
        a = jacobi_params[non_arcsine, 0]
        b = jacobi_params[non_arcsine, 1]

        mode = (b - a) / (a + b + 1)

        if log_scale:
            log_square_norm_P_0 =\
                (a + b + 1) * np.log(2) + betaln(a + 1, b + 1)
            bounds[0, non_arcsine] =\
                np.log(np.pi)\
                + (0.5 + a) * np.log(1 - mode)\
                + (0.5 + b) * np.log(1 + mode)\
                - log_square_norm_P_0
        else:
            square_norm_P_0 = 2**(a + b + 1) * beta(a + 1, b + 1)
            bounds[0, non_arcsine] =\
                np.pi\
                * (1 - mode)**(0.5 + a)\
                * (1 + mode)**(0.5 + b)\
                / square_norm_P_0

        # bounds[1:, non_arcsine] =
        #   2 * Gamma(n + 1 + a + b) Gamma(n + 1 + max(a,b))
        #   n! * (n+(a+b+1)/2)^(2 * max(a,b)) * Gamma(n + 1 + min(a,b))
        min_a_b = np.minimum(a, b)
        max_a_b = np.maximum(a, b)

        n = np.arange(1, deg_max + 1)[:, None]

        if log_scale:
            bounds[1:, non_arcsine] =\
                np.log(2)\
                + gammaln(n + 1 + a + b)\
                + gammaln(n + 1 + max_a_b)\
                - gammaln(n + 1)\
                - 2 * max_a_b * np.log(n + 0.5 * (a + b + 1))\
                - gammaln(n + 1 + min_a_b)
        else:
            bounds[1:, non_arcsine] =\
                2\
                * gamma(n + 1 + a + b)\
                * gamma(n + 1 + max_a_b)\
                / factorial(n)\
                / (n + 0.5 * (a + b + 1))**(2 * max_a_b)\
                / gamma(n + 1 + min_a_b)

    if log_scale:
        return np.exp(logsumexp(np.sum(bounds[ordering, range(dim)], axis=1)))
    else:
        return np.sum(np.prod(bounds[ordering, range(dim)], axis=1))


def poly_degrees(max_degrees):
    """ poly_degrees[i, j] = i if i <= max_degrees[j] else 0
    """

    max_deg, dim = max(max_degrees), len(max_degrees)
    polys = np.arange(max_deg + 1)[:, None] * np.ones(dim, dtype=int)
    polys[polys > max_degrees] = 0

    return max_deg, polys
