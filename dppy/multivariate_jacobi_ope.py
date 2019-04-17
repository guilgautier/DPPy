import numpy as np
import itertools as itt

from scipy.special import gamma, factorial, eval_jacobi


class MultivariateJacobiOPE:
    """
    Multivariate Jacobi Orthogonal Polynomial Ensemble used by :cite:`BaHa16` for Monte Carlo with Determinantal Point Processes.

    Reference weight
    .. math::

        w(x) = \prod_{i=1}^{d} (1-x)^{a_i} (1+x)^{b_i}

    :param N:
        Number of points :py:attr:`N`:math:`\geq 2`
    :type N:
        int, (default 10)

    :param jacobi_params:
        Jacobi parameters associated to
    :type N:
        array_like

    .. seealso::

        - :cite:`Kon05`
        - :cite:`BaHa16`
    """

    def __init__(self, N, jacobi_params):

        self.N, self.jacobi_params = self._check_params(N, jacobi_params)
        self.d = jacobi_params.shape[0]  # dimension

        self.max_layer_order = np.ceil(self.N**(1. / self.d)).astype(np.int16)
        self.ordering = self._compute_ordering()

        self.square_norms, self.rejection_bound =\
            self._compute_square_norms_and_rejection_bound()

    def _check_params(self, N, jacobi_params):
        """ Check that:

        - The number of points :math:`N \geq 2`
        - Jacobi parameters :math:`(a_i, b_i)_{i=1}^d \in [-0.5, 0.5]^d`.
        """
        if type(N) is not int or N < 2:
            return TypeError('Number of points N={} is not an integer or < 2'.format(N))

        if jacobi_params.ndim < 2:
            err = ('`jacobi_params.ndim = {}` < 2'.format(jacobi_params.ndim),
                'This class implements d-dimensional (d>1) Jacobi ensemble.',
                'For a 3-dimensional example please provide',
                '`jacobi_params` = array([[a_1, b_1], [a_2, b_2], [a_3, b_3]]',
                'For d=1, use the tridiagonal model, cf JacobiEnsemble!')
            return ValueError('\n'.join(err))

        if (-0.5 <= jacobi_params).all() and (jacobi_params <= 0.5).all():
            return N, jacobi_params
        else:
            raise ValueError('Jacobi parameters not in [-0.5, 0.5]^d, we have no guaranty')

    def _compute_ordering(self):
        """
        .. seealso::

            - :cite:`BaHa16` Section 2.1.3
        """
        ordering = itt.chain.from_iterable(
                    filter(lambda x: m - 1 in x,
                           itt.product(range(m), repeat=self.d))
                    for m in range(1, self.max_layer_order + 1))

        return list(ordering)[:self.N]

    def _compute_square_norms_and_rejection_bound(self):
        """

        Bound
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

            - :math:`\left\| P_n^{a,b} \right\|^2`, see    https://en.wikipedia.org/wiki/Jacobi_polynomials#Orthogonality
            - :cite:`Gau09` for the domination
        """

        # Initialize
        # - [square_norms]_ij = ||P_i^{a_j, b_j}||^2
        # - [bounds]_ij on
        #       pi (1-x)^(a_j+1/2) (1+x)^(b_j+1/2) P_i^2/||P_i||^2
        square_norms = np.zeros((self.max_layer_order, self.d))
        bounds = np.ones_like(square_norms)

        # range of the degrees of polynomials i = 0 .. max_layer_order
        rang = np.arange(self.max_layer_order)[:, None]

        arcsine = np.all(self.jacobi_params == -0.5, axis=1)
        if any(arcsine):
            # |P_0|^2 = pi
            # |P_n|^2 = 1/2 (Gamma(n+1/2)/n!)^2 otherwise
            square_norms[0, arcsine] = np.pi
            square_norms[1:, arcsine] =\
                0.5 * (gamma(0.5 + rang[1:]) / factorial(rang[1:]))**2

            # bounds[arcsine, 0] = pi / |P_0|^2 = 1, cf initialization
            bounds[1:, arcsine] = 2.0

        non_arcsine = np.any(self.jacobi_params != -0.5, axis=1)
        if any(non_arcsine):
            # |P_n|^2 = 2^(a + b + 1)      Gamma(n + 1 + a) Gamma(n + 1 + b)
            #           n! (2n + a + b + 1)     Gamma(n + 1 + a + b)

            a = self.jacobi_params[non_arcsine, 0]
            b = self.jacobi_params[non_arcsine, 1]

            square_norms[:, non_arcsine] =\
                gamma(rang + 1 + a)\
                * gamma(rang + 1 + b)\
                / gamma(rang + 1 + a + b)\
                / (2 * rang + 1 + a + b)\
                / factorial(rang)\
                * 2**(a + b + 1)

            # bounds[non_arcsine, 0]
            #   = pi * (1-mode)^(a+1/2) (1+mode)^(b+1/2) * 1 / ||P_0||^2
            # where mode = argmax (1-x)^(a+1/2) (1+x)^(b+1/2) = (b-a)/(a+b+1)
            mode = (b - a) / (a + b + 1)

            bounds[0, non_arcsine] =\
                np.pi\
                * (1 - mode)**(0.5 + a)\
                * (1 + mode)**(0.5 + b)\
                / square_norms[0, non_arcsine]

            # bounds[1:, non_arcsine] =
            # 2 * Gamma(n + 1 + a + b) Gamma(n + 1 + max(a,b))
            # n! * (n+(a+b+1)/2)^(2 * max(a,b)) * Gamma(n + 1 + min(a,b))
            min_a_b, max_a_b = np.minimum(a, b), np.maximum(a, b)

            bounds[1:, non_arcsine] =\
                gamma(rang[1:] + 1 + a + b)\
                * gamma(rang[1:] + 1 + max_a_b)\
                / gamma(rang[1:] + 1 + min_a_b)\
                / (rang[1:] + 0.5 * (a + b + 1))**(2 * max_a_b)\
                / factorial(rang[1:])\
                * 2

        # |P_alpha|^2 = \prod_{i=1}^d |P_{alpha_i}^{a_i,b_i}|^2
        multi_poly_sq_norm =\
            np.prod(square_norms[self.ordering, range(self.d)], axis=1)

        # Rejection bound
        # 1/(N-|Y|) (K(x,x) - K(x,Y) K(Y,Y)^-1 K(Y,x)) w(x)/w_eq
        # <= 1/(N-|Y|) K(x,x) w(x)/w_eq
        # <= rejection_bound / (N-|Y|)
        rejection_bound =\
            np.sum(np.prod(bounds[self.ordering, range(self.d)], axis=1))

        return multi_poly_sq_norm, rejection_bound

    def K(self, X, Y=None):
        '''
        .. math::
            K(x, y) = \sum_{\alpha}
                        \frac{P_{\alpha}(x)P_{\alpha}(y)}
                             {\|P_{\alpha}\|^2}

        for :math:`P_{\alpha}(x) = \prod_{i=1}^d P_{\alpha_i}^{a_i, b_i}(x_i)`
        '''

        if X.ndim == 1:
            if Y is None:
                return np.sum(
                            np.prod(
                                eval_jacobi(self.ordering,
                                            self.jacobi_params[:, 0],
                                            self.jacobi_params[:, 1],
                                            X),
                                axis=1)**2
                            /self.square_norms)

            elif Y.ndim == 1:
                return np.sum(
                            np.prod(
                                eval_jacobi(self.ordering,
                                            self.jacobi_params[:, 0],
                                            self.jacobi_params[:, 1],
                                            [X[None, :], Y[None, :]]),
                                axis=(0, 2)
                                ) / self.square_norms
                            )

            else:  # [K(X, Y_1), ..., K(X, Y_n)]

                pol_evals = np.prod(
                                eval_jacobi(self.ordering,
                                            self.jacobi_params[:, 0],
                                            self.jacobi_params[:, 1],
                                            np.vstack((Y, X))[:, None]),
                                axis=2)

                return np.sum(
                            (pol_evals[-1] * pol_evals[:-1])\
                                / self.square_norms,
                            axis=1)

        else:
            print('Weird call of this function X.shape={}, Y.shape={}'.format(X.shape, Y.shape))

    def sample_from_proposal(self, a=0.5, b=0.5):

        return 2.0 * np.random.beta(a, b, size=self.d) - 1.0

    def sample(self, nb_trials_max=2000):
        """ Rejection sampling with proposal :math:`\mu_{\text{eq}}^{\otimes d}` where
        .. math::

            \mu_{\text{eq}}
                = \frac{1}{\pi \sqrt{1-x^2}} 1_{(-1, 1)}(x) d x

        """

        sample = np.zeros((self.N, self.d))
        K_xY = np.zeros(self.N)
        K_1 = np.zeros((self.N, self.N))

        nb_reject = 0

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
                nb_reject += 1
                sample[it] = 1e-5 * np.random.randn(self.d)

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
                                        [-K_xY[:it], K_YY]])\
                                / (K_xY[it] * K_YY - K_xY[:it]**2)

            else:
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

        print('nb_rejection', nb_reject)
        return sample

    def eval_w_over_mu_eq(self, x):

        a_b = 0.5 + self.jacobi_params

        return np.pi**self.d * np.prod((1 - x)**a_b[:, 0] * (1 + x)**a_b[:, 1])
