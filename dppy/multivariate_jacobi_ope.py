# coding: utf8
""" Implementation of the class :class:`MultivariateJacobiOPE` which has 3 main methods:

- :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.sample` to get a sample of
- :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.K` to evaluate the corresponding projection kernel
- :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.plot` to display 1D or 2D samples
"""

import numpy as np
import itertools as itt

from scipy import stats
from scipy.special import beta, betaln, factorial, gamma, gammaln
from scipy.special import eval_jacobi
# from scipy.special import logsumexp

from  dppy.random_matrices import mu_ref_beta_sampler_tridiag as tridiagonal_model

from sys import platform as _platform
# https://stackoverflow.com/questions/1854/python-what-os-am-i-running-on
if _platform.startswith('linux'):
    # linux
    pass
elif _platform == "darwin":
    # MAC OS X
    # https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/
    # import matplotlib
    # matplotlib.use('TkAgg')
    pass

import matplotlib.pyplot as plt

from dppy.utils import check_random_state


class MultivariateJacobiOPE:
    """
    Multivariate Jacobi Orthogonal Polynomial Ensemble used by :cite:`BaHa16` for Monte Carlo with Determinantal Point Processes and in our `ICML'19 workshop paper <http://negative-dependence-in-ml-workshop.lids.mit.edu/wp-content/uploads/sites/29/2019/06/icml_camera_ready.pdf>`_

    .. important::

        In the current implementation of the chain rule, the proposal density is :math:`\\frac{1}{N} K(x,x) w(x)` and not :math:`\\prod_{i=1}^d \\frac{1}{\\pi\\sqrt{1-(x^i)^2}}` as previously used in the references mentioned above.

        This yields faster sampling since less evaluations of the conditionnals involving Schur complements are required, see also :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.sample`

    The multivariate Jacobi orthogonal polynomial ensemble corresponds to a continuous multivariate projection DPP with state space :math:`[-1, 1]^d` and

    - reference measure :math:`\\mu(dx) = w(x) dx` (see also :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.eval_w`), where

        .. math::

            w(x) = \\prod_{i=1}^{d} (1-x)^{a_i} (1+x)^{b_i}

    - kernel :math:`K` (see also :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.K`)

        .. math::
            K(x, y) = \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                        \\frac{P_{k}(x)P_{k}(y)}
                              {\\|P_{k}\\|^2}

        where

        - :math:`k \\in \\mathbb{N}^d` is a multi-index ordered according to the ordering :math:`\\mathfrak{b}` (see :py:meth:`compute_ordering_BaHa16`)

        - :math:`P_{k}(x) = \\prod_{i=1}^d P_{k_i}^{a_i, b_i}(x_i)` is the product of orthogonal Jacobi polynomials w.r.t. :math:`\\mu`, (see :py:meth:`eval_poly_multiD`)

            .. math::

                \\int_{-1}^{1}
                    P_{k}^{(a_i,b_i)}(x) P_{\\ell}^{(a_i,b_i)}(x)
                    (1-x)^{a_i} (1+x)^{b_i} d x
                \\propto \\delta_{k\\ell}

    :param N:
        Number of points :math:`N \\geq 2`
    :type N:
        int

    :param jacobi_params:
        Jacobi parameters :math:`[(a_i, b_i)]_{i=1}^d \\in [-\\frac{1}{2}, \\frac{1}{2}]^{d \\times 2}`.
        The number of rows :math:`d` prescribes the ambient dimension of the points i.e. :math:`x_{1}, \\dots, x_{N} \\in [-1, 1]^d`
    :type jacobi_params:
        array_like

    .. seealso::

        - :ref:`multivariate_jacobi_ope`
        - Univariate :ref:`jacobi_banded_matrix_model` sampled with the tridiagonal model of :cite:`KiNe04`
        - :cite:`BaHa16` initiated the use of the multivariate Jacobi ensemble for Monte Carlo integration. In particular, they proved a fast CLT with variance decay of order :math:`N^{1+1/d}`
    """

    def __init__(self, N, jacobi_params):

        self.N, self.jacobi_params, self.dim =\
            self._check_params(N, jacobi_params)

        self.ordering = compute_ordering_BaHa16(self.N, self.dim)

        self.deg_max, self.poly_1D_degrees =\
            poly_1D_degrees(np.max(self.ordering, axis=0))

        self.poly_1D_square_norms =\
            compute_poly1D_square_norms(self.jacobi_params, self.deg_max)

        self.poly_multiD_square_norms =\
            np.prod(self.poly_1D_square_norms[self.ordering, range(self.dim)],
                    axis=1)

        self.mass_of_mu = self.poly_multiD_square_norms[0]

        self.Gautschi_bounds =\
            compute_Gautschi_bounds(self.jacobi_params,
                                    self.ordering,
                                    log_scale=True)

        self._jacobi_params_plus_05 = 0.5 + self.jacobi_params
        self._pi_power_dim = np.pi**self.dim

    def _check_params(self, N, jacobi_params):
        """ Check that:

        - The number of points :math:`N \\geq 2`
        - Jacobi parameters :math:`[(a_i, b_i)]_{i=1}^d \\in [-0.5, 0.5]^{d\\times 2}`.
        """
        if type(N) is not int or N < 1:
            return TypeError('Number of points N={} is not an integer or < 2'.format(N))

        dim = jacobi_params.size // 2
        # if dim == 1:
        #     war = 'In dimension {}, the tridiagonal model is used'.format(dim)
        #     warnings.warn(war)

        if np.all(np.abs(jacobi_params) <= 0.5):
            return N, jacobi_params, dim
        else:
            raise ValueError('Jacobi parameters not in [-0.5, 0.5]^d, we have no guaranty')

    def eval_w(self, X, jac_params=None):
        """Evaluate :math:`w(x) = \\prod_{i=1}^{d} (1-x^i)^{a_i} (1+x^i)^{b_i}` which corresponds to the density of the base measure :math:`\\mu` if ``jac_params`` is ``None``.

        :param X:
            Array of points :math:`\\in [-1, 1]^d`, with size :math:`n\\times d` where :math:`n` is the number of points
        :type X:
            array_like

        :param jac_params:
            - if ``None``, use attribute ``MultivariateJacobiOPE.jacobi_params`` i.e. :math:`[(a_i, b_i)]_{i=1}^d \\in [-0.5, 0.5]^{d \\times 2}`.
            - else Jacobi parameters :math:`[(a_i, b_i)]_{i=1}^d \\in [-1, \\infty)^{d \\times 2}`.
        :type jac_params:
            array_like (default None)

        :return:
            - if ``jac_params`` is ``None``, evaluation of :math:`w(x)` the density of the base measure
            - else ``jac_params`` :math:`= [(a_i, b_i)]_{i=1}^d` evaluation of :math:`w(x) = \\prod_{i=1}^{d} (1-x^i)^{a_i} (1+x^i)^{b_i}`
        :rtype:
            array_like
        """
        if jac_params is None:
            return np.prod((1.0 - X)**(self.jacobi_params[:, 0])
                         * (1.0 + X)**(self.jacobi_params[:, 1]),
                            axis=-1)
        else:
            if np.all(jac_params > -1):
                return np.prod((1.0 - X)**(jac_params[:, 0])
                             * (1.0 + X)**(jac_params[:, 1]),
                                axis=-1)
            else:
                raise ValueError('Invalid Jacobi parameters, must be all > -1')

    def K(self, X, Y=None):
        '''Evaluate the orthogonal projection kernel :math:`K`.
        It is based on the `3-terms recurrence relations <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>`_ satisfied by each univariate orthogonal Jacobi polynomial, `see also SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_jacobi.html>`_ :func:`eval_jacobi`

        .. math::
            K(x, y) = \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                        \\frac{P_{k}(x)P_{k}(y)}
                              {\\|P_{k}\\|^2}

        where

        - :math:`k \\in \\mathbb{N}^d` is a multi-index ordered according to the ordering :math:`\\mathfrak{b}`, :py:meth:`compute_ordering_BaHa16`

        - :math:`P_{k}(x) = \\prod_{i=1}^d P_{k_i}^{a_i, b_i}(x^i)` is the product of orthogonal Jacobi polynomials w.r.t. :math:`\\mu(dx) = \\prod_{i=1}^{d} (1-x^i)^{a_i} (1+x^i)^{b_i} d x^i`

            .. math::

                \\int_{-1}^{1}
                    P_{k}^{(a_i,b_i)}(x) P_{\\ell}^{(a_i,b_i)}(x)
                    (1-x)^{a_i} (1+x)^{b_i} d x
                \\propto \\delta_{k\\ell}

        :param X:
            Array of points :math:`\\in [-1, 1]^d`, with size :math:`n\\times d` where :math:`n` is the number of points
        :type X:
            array_like

        :param Y:
            Array of points :math:`\\in [-1, 1]^d`, with size :math:`n\\times d` where :math:`n` is the number of points
        :type Y:
            array_like (default None)

        :return:
            Pointwise evaluation of :math:`K` as depicted in the following pseudo code output

            - if ``Y`` is ``None``

                - ``K(X, X)`` if ``X.size`` :math:`=d`
                - ``[K(x, x) for x in X]`` otherwise

            - otherwise

                - ``K(X, Y)`` if ``X.size=Y.size``:math:`=d`
                - ``[K(X, y) for y in Y]`` if ``X.size`` :math:`=d`
                - ``[K(x, y) for x, y in zip(X, Y)]`` otherwise
        :rtype:
            - float if ``Y`` is ``None`` and ``X.size`` :math:`=d`
            - array_like otherwise
        '''
        if Y is None:

            if X.size == self.dim:  # X is vector in R^d
                polys_X_2 = eval_jacobi(self.poly_1D_degrees,
                                        self.jacobi_params[:, 0],
                                        self.jacobi_params[:, 1],
                                        X)**2\
                            / self.poly_1D_square_norms

                return np.sum(
                            np.prod(
                                polys_X_2[self.ordering, range(self.dim)],
                            axis=1),
                        axis=0)

            else:
                polys_X_2 = eval_jacobi(self.poly_1D_degrees,
                                        self.jacobi_params[:, 0],
                                        self.jacobi_params[:, 1],
                                        X[:, None])**2\
                            / self.poly_1D_square_norms

                return np.sum(
                            np.prod(
                                polys_X_2[:, self.ordering, range(self.dim)],
                            axis=2),
                        axis=1)


        else:

            lenX = X.size // self.dim  # X.shape[0] if X.ndim > 1 else 1
            lenY = Y.size // self.dim  # Y.shape[0] if Y.ndim > 1 else 1

            polys_X_Y = eval_jacobi(self.poly_1D_degrees,
                                    self.jacobi_params[:, 0],
                                    self.jacobi_params[:, 1],
                                    np.vstack((X, Y))[:, None])

            if lenX > lenY:

                polys_X_Y[:lenX] *= polys_X_Y[lenX:] / self.poly_1D_square_norms

                return np.sum(
                            np.prod(
                                polys_X_Y[:lenX, self.ordering, range(self.dim)],
                            axis=2),
                        axis=1)

            else:  # if lenX < lenY:

                polys_X_Y[lenX:] *= polys_X_Y[:lenX] / self.poly_1D_square_norms

                return np.sum(
                            np.prod(
                                polys_X_Y[lenX:, self.ordering, range(self.dim)],
                            axis=2),
                        axis=1)

    def sample_proposal_lev_score(self, nb_trials_max=10000,
                                  random_state=None):
        """Use a rejection sampling mechanism to sample

        .. math::

            \\frac{1}{N} K(x, x) w(x) dx
            = \\frac{1}{N} \\sum_{\\mathfrak{k}=0}^{N-1} P_k(x)^2 w(x)

        with proposal distribution

        .. math::

            w_{eq}(x) d x
            = \\frac{1}{\\pi^d} \\prod_{i=1}^{d} \\frac{1}{\\sqrt{1-(x^i)^2}} d x

        Since the target density is a mixture, we can sample it by

        1. Draw a multi-index :math:`k` uniformly at random
        2. Sample from :math:`P_k(x)^2 w(x) dx` with proposal :math:`w_{eq}(x) d x`.

            The acceptance ratio writes

            .. math::

                P_k(x)^2 w(x) \\frac{1}{\\frac{w_{\\text{eq}}(x)}{\\pi^d}}
                = \\prod_{i=1}^{d} \\pi P_{k^i}(x)^2 (1-x^i)^{a^i+\\frac{1}{2}} (1+x^i)^{b^i+\\frac{1}{2}}

            which can be bounded using the result of :cite:`Gau09` on Jacobi polynomials.
            It is computed at initialization of the :py:class:`MultivariateJacobiOPE` object with :py:meth:`compute_Gautschi_bounds`

        :return:
            A sample :math:`x\\in[-1,1]^d` with probability distribution :math:`\\frac{1}{N} K(x,x) w(x)`
        :rtype:
            array_like

        .. seealso::

            - :py:meth:`compute_Gautschi_bounds`
            - :py:meth:`sample`
        """
        rng = check_random_state(random_state)

        ind = rng.randint(self.N)
        k_multi_ind = self.ordering[ind]
        square_norm = self.poly_multiD_square_norms[ind]
        Gautschi_bound = self.Gautschi_bounds[ind]

        for trial in range(nb_trials_max):

            # Propose x ~ arcsine = \prod_{i=1}^{d} 1/pi 1/sqrt(1-(x^i)^2)
            x = 1.0 - 2.0 * rng.beta(0.5, 0.5, size=self.dim)

            Pk2_x = np.prod(eval_jacobi(k_multi_ind,
                                        self.jacobi_params[:, 0],
                                        self.jacobi_params[:, 1],
                                        x))**2
            Pk2_x /= square_norm

            ratio_w_proposal =\
                self._pi_power_dim\
                * np.prod((1.0 - x)**(self._jacobi_params_plus_05[:, 0])
                        * (1.0 + x)**(self._jacobi_params_plus_05[:, 1]))

            if rng.rand() * Gautschi_bound < Pk2_x * ratio_w_proposal:
                break
        else:
            print('proposal not enough trials')

        return x, self.K(x, None)

    def eval_poly_multiD(self, X, normalize='norm'):
        """Evaluate (and potentially normalize) multivariate Jacobi polynomials :math:`P_{k}(x) = \\prod_{i=1}^d P_{k_i}^{a_i, b_i}(x_i)`

        :param X:
            Array of points :math:`\\in [-1, 1]^d`, with size :math:`n\\times d` where :math:`n` is the number of points
        :type X:
            array_like

        :param normalize:
            - 'norm'
            - 'square_norm'
        :type normalize:
            str (default 'norm')

        :return:
            - ``normalize='norm'`` :math:`P_k(X) / \\left\\| P_k \\right\\|`
            - ``normalize='square_norm'`` :math:`P_k(X) / \\left\\| P_k \\right\\|^2`
        :rtype:
            array_like

        .. seealso::

            - evaluation of the kernel :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.K`
        """

        if X.size == self.dim:
            poly_1D_jacobi = eval_jacobi(self.poly_1D_degrees,
                                         self.jacobi_params[:, 0],
                                         self.jacobi_params[:, 1],
                                         X)

            if normalize == 'square_norm':
                poly_1D_jacobi /= self.poly_1D_square_norms

            elif normalize == 'norm':
                poly_1D_jacobi /= np.sqrt(self.poly_1D_square_norms)

            # else:
            #     pass

            return np.prod(poly_1D_jacobi[self.ordering, range(self.dim)],
                           axis=1)

        else:
            poly_1D_jacobi = eval_jacobi(self.poly_1D_degrees,
                                         self.jacobi_params[:, 0],
                                         self.jacobi_params[:, 1],
                                         X[:, None])

            if normalize == 'square_norm':
                poly_1D_jacobi /= self.poly_1D_square_norms

            elif normalize == 'norm':
                poly_1D_jacobi /= np.sqrt(self.poly_1D_square_norms)

            # else:
            #     pass

            return np.prod(poly_1D_jacobi[:, self.ordering, range(self.dim)],
                           axis=2)

    def sample(self, nb_trials_max=10000, random_state=None):
        """Use the chain rule :cite:`HKPV06` (Algorithm 18) to sample :math:`\\left(x_{1}, \\dots, x_{N} \\right)` with density

        .. math::

            & \\frac{1}{N!}
                \\left[K(x_n,x_p)\\right]_{n,p=1}^{N}
                \\prod_{n=1}^{N} w(x_n)\\\\
            &= \\frac{1}{N} K(x_1,x_1) w(x_1)
            \\prod_{n=2}^{N}
                \\frac{
                    K(x_n,x_n)
                    - K(x_n,x_{1:n-1})
                    \\left[\\left[K(x_k,x_l)\\right]_{k,l=1}^{n-1}\\right]^{-1}
                    K(x_{1:n-1},x_n)
                    }{N-(n-1)}
                    w(x_n)

        The order in which the points were sampled can be forgotten to obtain a valid sample of the corresponding DPP

        Each conditional density is sampled using rejection sampling with proposal density :math:`\\frac{1}{N} K(x,x) w(x)`

        - :math:`x_1 \\sim \\frac{1}{N} K(x,x) w(x)` using :py:meth:`sample_proposal_lev_score`

        - :math:`x_n | Y = x_{1}, \\dots, x_{n-1}`

            .. math::

                \\frac{1}{N-|Y|} [K(x,x) - K(x, Y) K_Y^{-1} K(Y, x)] w(x)
                \\leq \\frac{N}{N-|Y|} \\frac{1}{N} K(x,x) w(x)

        .. seealso::

            - :ref:`finite_dpps_exact_sampling_projection_dpp_chain_rule`
            - :py:meth:`sample_proposal_lev_score`
        """

        rng = check_random_state(random_state)

        if self.dim == 1:
            sample = tridiagonal_model(a=self.jacobi_params[0, 0] + 1,
                                       b=self.jacobi_params[0, 1] + 1,
                                       size=self.N,
                                       random_state=rng)[:, None]
            return 1.0 - 2.0 * sample

        # In multi D
        sample = np.zeros((self.N, self.dim))
        # To compute Schur complement
        # schur = K(x, x) - K(x, Y) K(Y, Y)^-1 K(Y, x)
        #
        # K(x, x) = K_xx above
        sample[0], K_xx = self.sample_proposal_lev_score(random_state=rng)
        # K_YY^-1 = K(Y, Y)^-1
        K_Y_inv = np.zeros((self.N - 1, self.N - 1))
        K_Y_inv[0, 0] = 1.0 / K_xx
        # K_Yx = Phi_Y.dot(Phi_x) = Phi_Yx[:it].dot(Phi_Yx[it])
        K_Yx = np.zeros(self.N - 1)
        Phi_Yx = np.zeros((self.N, self.N))
        Phi_Yx[0] = self.eval_poly_multiD(sample[0], normalize='square_norm')
        temp = np.zeros(self.N - 1)

        for it in range(1, self.N):

            for trial in range(nb_trials_max):

                # Propose a point from 1/N K(x,x) w(x) i.e. leverage score
                sample[it], K_xx = self.sample_proposal_lev_score(random_state=rng)

                # Compute Schur cmplmt = K(x, x) - K(x, Y) K(Y, Y)^-1 K(Y, x)
                #
                # K_Yx = Phi_Y.dot(Phi_x)
                Phi_Yx[it] = self.eval_poly_multiD(sample[it], normalize='')
                K_Yx[:it] = Phi_Yx[:it].dot(Phi_Yx[it])
                # Schur complement
                schur = K_xx - K_Yx[:it].dot(K_Y_inv[:it, :it]).dot(K_Yx[:it])

                if rng.rand() < schur / K_xx:
                    break
            else:
                print('chain rule iteration {} not enough trials'.format(it))

            if it < self.N - 1:
                # Normalize Phi_x with square norm
                Phi_Yx[it] /= self.poly_multiD_square_norms
                # Update [K_Y]^-1
                # K_{Y+x}^-1 =
                # [[K_Y^-1 + (K_Y^-1 K_Yx K_xY K_Y^-1)/schur(x),
                #   -K_Y^-1 K_Yx / schur(x)],
                # [-K_xY K_Y^-1/ schur(x),
                #   1/schur(x)]]
                temp[:it] = K_Y_inv[:it, :it].dot(K_Yx[:it])

                K_Y_inv[:it, :it] += np.outer(temp[:it], temp[:it] / schur)
                K_Y_inv[:it, it] = - temp[:it] / schur
                K_Y_inv[it, :it] = K_Y_inv[:it, it]

                K_Y_inv[it, it] = 1.0 / schur

        return sample

    def plot(self, sample=None, weighted=False):

        if self.dim >= 3:
            raise NotImplementedError(
                'Visualizations in dimension >= 3 are not implemented')

        tols = np.zeros_like(self.jacobi_params)
        tols[tols < 0] = 5e-2

        if sample is None:
            sample = self.sample()

        weights = 1. / self.K(sample) if weighted else None

        ticks_pos = [-1, 0, 1]
        ticks_labs = list(map(str, ticks_pos))

        if self.dim == 1:

            fig, ax_main = plt.subplots(figsize=(10, 8))

            ax_main.tick_params(axis='both', which='major', labelsize=18)
            ax_main.set_xticks(ticks_pos)
            ax_main.set_xticklabels(ticks_labs)

            ax_main.spines['right'].set_visible(False)
            ax_main.spines['top'].set_visible(False)

            ax_main.scatter(sample[:, 0],
                            np.zeros_like(sample[:, 0]),
                            s=2000 * weights if weighted else 1)

            ax_main.hist(sample[:, 0],
                         bins=10,
                         weights=weights,
                         density=True,
                         orientation='vertical',
                         alpha=0.5)

            # Top densities
            X_ = np.linspace(-1 + tols[0, 1], 1 - tols[0, 0], 200)[:, None]
            ax_main.plot(X_,
                         0.5 * stats.beta(*(1 + self.jacobi_params[0])).pdf(0.5 * (1 - X_)),
                         ls='--', c='red', lw=3, alpha=0.7,
                         label=r'$a_1 = {:.2f}, b_1 = {:.2f}$'.format(*self.jacobi_params[0]))

            x_lim = ax_main.get_xlim()
            y_lim = ax_main.get_ylim()

            if not weighted:

                tol = 5e-2
                X_ = np.linspace(-1 + tol, 1 - tol, 200)[:, None]
                ax_main.plot(X_,
                             0.5 * stats.beta(0.5, 0.5).pdf(0.5 * (1 - X_)),
                             c='orange', ls='-', lw=3,
                             label=r'$a = b = -0.5$')

            ax_main.legend(fontsize=20,
                           loc='center',
                           bbox_to_anchor=(0.5, -0.1 if weighted else -0.12),
                           labelspacing=0.1,
                           frameon=False)

        elif self.dim == 2:

            # Create Fig and gridspec
            fig = plt.figure(figsize=(8, 8))
            grid = plt.GridSpec(6, 6, hspace=0., wspace=0.)

            ax_main = fig.add_subplot(grid[1:, :-1],
                                      xticks=ticks_pos, xticklabels=ticks_labs,
                                      yticks=ticks_pos, yticklabels=ticks_labs)

            ax_main.tick_params(axis='both', which='major', labelsize=18)

            ax_main.scatter(sample[:, 0],
                            sample[:, 1],
                            s=2000 * weights if weighted else 10)

            x_lim = ax_main.get_xlim()
            y_lim = ax_main.get_ylim()

            # Top plot
            ax_top = fig.add_subplot(grid[0, :-1],
                                     xticks=ticks_pos, xticklabels=[],
                                     yticks=[], yticklabels=[],
                                     frameon=False)
            ax_top.set_xlim(x_lim)

            # Top histogram
            ax_top.hist(sample[:, 0],
                        bins=10,
                        weights=weights,
                        density=True,
                        orientation='vertical',
                        alpha=0.5)

            # Top densities
            X_ = np.linspace(-1 + tols[0, 1], 1 - tols[0, 0], 200)[:, None]
            l_top, = ax_top.plot(X_,
                                 0.5 * stats.beta(*(1 + self.jacobi_params[0])).pdf(0.5 * (1 - X_)),
                                 ls='--', c='red', lw=3, alpha=0.7)

            # Right plot
            ax_right = fig.add_subplot(grid[1:, -1],
                                       xticks=[], xticklabels=[],
                                       yticks=ticks_pos, yticklabels=[],
                                       frameon=False)
            ax_right.set_ylim(y_lim)

            # Right histogram
            ax_right.hist(sample[:, 1],
                          bins=10,
                          weights=weights,
                          density=True,
                          orientation='horizontal',
                          alpha=0.5)

            # Right densities
            X_ = np.linspace(-1 + tols[1, 1], 1 - tols[1, 0], 200)[:, None]
            l_right, = ax_right.plot(0.5 * stats.beta(*(1 + self.jacobi_params[1])).pdf(0.5 * (1 - X_)),
                                     X_,
                                     ls='--', c='green', lw=3, alpha=0.7)

            leg_axes = [l_top, l_right]
            leg_text = [', '.join([r'$a_{} = {:.2f}$'.format(i+1, jac_par[0]),
                                   r'$b_{} = {:.2f}$'.format(i+1, jac_par[1])])
                        for i, jac_par in enumerate(self.jacobi_params)]

            if not weighted:

                tol = 5e-2
                X_ = np.linspace(-1 + tol, 1 - tol, 200)[:, None]
                l_arcsine, = ax_top.plot(X_,
                                         0.5 * stats.beta(0.5, 0.5).pdf(0.5 * (1 - X_)),
                                         c='orange', ls='-', lw=3)
                ax_right.plot(0.5 * stats.beta(0.5, 0.5).pdf(0.5 * (1 - X_)),
                              X_,
                              c='orange', ls='-', lw=3)

                leg_axes.append(l_arcsine)
                leg_text.append(r'$a = b = -0.5$')

            ax_main.legend(leg_axes,
                           leg_text,
                           fontsize=20,
                           loc='center',
                           bbox_to_anchor=(0.5, -0.14 if weighted else -0.18),
                           labelspacing=0.1,
                           frameon=False)


def compute_ordering_BaHa16(N, d):
    """ Compute the ordering of the multi-indices :math:`\\in\\mathbb{N}^d` defining the order between the multivariate monomials as described in Section 2.1.3 of :cite:`BaHa16`.

    :param N:
        Number of polynomials :math:`(P_k)` considered to build the kernel :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.K` (number of points of the corresponding :py:class:`MultivariateJacobiOPE`)
    :type N:
        int

    :param d:
        Size of the multi-indices :math:`k\\in \\mathbb{N}^d` characterizing the _degree_ of :math:`P_k` (ambient dimension of the points x_{1}, \\dots, x_{N} \\in [-1, 1]^d)
    :type d:
        int

    :return:
        Array of size :math:`N\\times d` containing the first :math:`N` multi-indices :math:`\\in\\mathbb{N}^d` in the order prescribed by the ordering :math:`\\mathfrak{b}` :cite:`BaHa16` Section 2.1.3
    :rtype:
        array_like

    For instance, for :math:`N=12, d=2`

    .. code:: python

        [(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2), (0, 3), (1, 3), (2, 3)]

    .. seealso::

        - :cite:`BaHa16` Section 2.1.3
    """
    layer_max = np.floor(N**(1.0 / d)).astype(np.int16)

    ordering = itt.chain.from_iterable(
                filter(lambda x: m in x, itt.product(range(m + 1), repeat=d))
                for m in range(layer_max + 1))

    return list(ordering)[:N]


def compute_poly1D_square_norms(jacobi_params, deg_max):
    """ Compute the square norms :math:`\\|P_{k}^{(a_i,b_i)}\\|^2` of each (univariate) Jacobi polynomial for :math:`k=0` to ``deg_max`` and :math:`a_i, b_i =` ``jacobi_params[i, :]``
    Recall that the Jacobi polynomials :math:`\\left( P_{k}^{(a_i,b_i)} \\right)` are `orthogonal <http://en.wikipedia.org/wiki/Jacobi_polynomials#Orthogonality>`_ w.r.t. :math:`(1-x)^{a_i} (1+x)^{b_i}`.

    .. math::

        \\|P_{k}^{(a_i,b_i)}\\|^2
            = \\int_{-1}^{1}
                \\left( P_{k}^{(a_i,b_i)}(x) \\right)^2
                (1-x)^{a_i} (1+x)^{b_i} d x

    :param jacobi_params:
        Jacobi parameters :math:`[(a_i, b_i)]_{i=1}^d \\in [-\\frac{1}{2}, \\frac{1}{2}]^{d \\times 2}`
        The number of rows :math:`d` prescribes the ambient dimension of the points i.e. :math:`x_{1}, \\dots, x_{N} \\in [-1, 1]^d`
    :type jacobi_params:
        array_like

    :param deg_max:
        Maximal degree of 1D Jacobi polynomials
    :type deg_max:
        int

    :return:
        Array of size ``deg_max + 1`` :math:`\\times d` with entry :math:`k,i` given by :math:`\\|P_{k}^{(a_i,b_i)}\\|^2`
    :rtype:
        array_like

    .. seealso::

        - `Wikipedia Jacobi polynomials <http://en.wikipedia.org/wiki/Jacobi_polynomials#Orthogonality>`_
        - :py:meth:`compute_ordering_BaHa16`
    """

    # Initialize
    # - [square_norms]_ij = ||P_i^{a_j, b_j}||^2
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


def compute_Gautschi_bounds(jacobi_params, ordering, log_scale=True):
    """ Compute the rejection constants for the acceptance/rejection mechanism used in :py:meth:`sample_proposal_lev_score` to sample

    .. math::

        \\frac{1}{N} K(x, x) w(x) dx
        = \\frac{1}{N} \\sum_{\\mathfrak{k}=0}^{N-1} P_k(x)^2 w(x)

    with proposal distribution

    .. math::

        w_{eq}(x) d x
        = \\frac{1}{\\pi^d} \\prod_{i=1}^{d} \\frac{1}{\\sqrt{1-(x^i)^2}} d x

    To get a sample:

    1. Draw a multi-index :math:`k` uniformly at random
    2. Sample from :math:`P_k(x)^2 w(x) dx` with proposal :math:`w_{eq}(x) d x`.

        The acceptance ratio writes

        .. math::

            P_k(x)^2 w(x) \\frac{1}{\\frac{w_{\\text{eq}}(x)}{\\pi^d}}
            = \\prod_{i=1}^{d} \\pi P_{k^i}(x)^2 (1-x^i)^{a^i+\\frac{1}{2}} (1+x^i)^{b^i+\\frac{1}{2}}


    - For :math:`k_i>0` we use a result on Jacobi polynomials given by, e.g., :cite:`Gau09`

    .. math::

        &
            \\pi
            (1-x)^{a+\\frac{1}{2}}
            (1+x)^{b+\\frac{1}{2}}
            \\frac{P_{n}^{(a, b)}(x)^2}{\\|P_{n}^{(a, b)}\\|^2}\\\\
        &\\leq
            \\frac{2}
                  {n!(n+(a+b+1) / 2)^{2 \\max(a,b)}}
            \\frac{\\Gamma(n+a+b+1)
                    \\Gamma(n+\\max(a,b)+1)}
                  {\\Gamma(n+\\min(a,b)+1)},
        \\quad|a|,|b| \\leq \\frac{1}{2}

    - For :math:`k_i=0`, we use less involved properties of the `Jacobi polynomials <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_:

        - :math:`P_{0}^{(a, b)} = 1`
        - :math:`\\|P_{0}^{(a, b)}\\|^2 = 2^{a+b+1} \\operatorname{B}(a+1,b+1)`
        - :math:`m = \\frac{b-a}{a+b+1}` is the mode of :math:`(1-x)^{a+\\frac{1}{2}} (1+x)^{b+\\frac{1}{2}}` (valid since :math:`a+\\frac{1}{2}, b+\\frac{1}{2} > 0`)

    So that,

    .. math::

            \\pi
            (1-x)^{a+\\frac{1}{2}}
            (1+x)^{b+\\frac{1}{2}}
            \\left[\\frac{P_{0}^{(a, b)}(x)}
                   {\\|P_{0}^{(a, b)}\\|}\\right]^{2}
        =
            \\frac
            {\\pi
             (1-x)^{a+\\frac{1}{2}}
             (1+x)^{b+\\frac{1}{2}}}
            {\\|P_{0}^{(a, b)}\\|^2}
        \\leq
            \\frac
            {\\pi
             (1-m)^{a+\\frac{1}{2}}
             (1+m)^{b+\\frac{1}{2}}}
            {2^{a+b+1} \\operatorname{B}(a+1,b+1)}

    :param jacobi_params:
        Jacobi parameters :math:`[(a_i, b_i)]_{i=1}^d \\in [-\\frac{1}{2}, \\frac{1}{2}]^{d \\times 2}`.

        The number of rows :math:`d` prescribes the ambient dimension of the points i.e. :math:`x_{1}, \\dots, x_{N} \\in [-1, 1]^d`
    :type jacobi_params:
        array_like

    :param ordering:
        Ordering of the multi-indices :math:`\\in\\mathbb{N}^d` defining the order between the multivariate monomials (see also :py:meth:`compute_ordering_BaHa16`)

        - the number of rows corresponds to the number :math:`N` of monomials considered.
        - the number of columns :math:`=d`

    :type ordering:
        array_like

    :param log_scale:
        Specify if, when computing the rejection bound, the logarithmic versions ``betaln``, ``gammaln`` of ``beta`` and ``gamma`` functions are used to avoid overflows
    :type log_scale:
        bool

    :return:
        Gautschi bound
    :rtype:
        float

    .. see also::

        - :cite:`Gau09` for the domination when :math:`k_i > 0`
        - :py:meth:`compute_poly1D_square_norms`
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
        return np.exp(np.sum(bounds[ordering, range(dim)], axis=1))
    else:
        return np.prod(bounds[ordering, range(dim)], axis=1)


def poly_1D_degrees(max_degrees):
    """ :return:poly_1D_degrees[i, j] = i if i <= max_degrees[j] else 0
    """

    max_deg, dim = max(max_degrees), len(max_degrees)
    polys = np.arange(max_deg + 1)[:, None] * np.ones(dim, dtype=int)
    polys[polys > max_degrees] = 0

    return max_deg, polys
