# coding: utf8
""" Implementation of the class :class:`MultivariateJacobiOPE` used in :cite:`GaBaVa19` for Monte Carlo with Determinantal Point Processes

It has 3 main methods:

- :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.sample` to generate samples
- :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.K` to evaluate the corresponding projection kernel
- :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.plot` to display 1D or 2D samples
"""

import numpy as np
import itertools as itt

from scipy import stats
from scipy.special import beta, betaln, factorial, gamma, gammaln
from scipy.special import eval_jacobi
# from scipy.special import logsumexp
import matplotlib.pyplot as plt

from dppy.random_matrices import mu_ref_beta_sampler_tridiag as tridiagonal_model

from dppy.utils import check_random_state, inner1d


class MultivariateJacobiOPE:
    """
    Multivariate Jacobi Orthogonal Polynomial Ensemble used in :cite:`GaBaVa19` for Monte Carlo with Determinantal Point Processes

    This corresponds to a continuous multivariate projection DPP with state space :math:`[-1, 1]^d` with respect to

    - reference measure :math:`\\mu(dx) = w(x) dx` (see also :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.eval_w`), where

        .. math::

            w(x) = \\prod_{i=1}^{d} (1-x_i)^{a_i} (1+x_i)^{b_i}

    - kernel :math:`K` (see also :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.K`)

        .. math::
            K(x, y) = \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                        P_{k}(x) P_{k}(y)
                    = \\Phi(x)^{\\top} \\Phi(y)

        where

        - :math:`k \\in \\mathbb{N}^d` is a multi-index ordered according to the ordering :math:`\\mathfrak{b}` (see :py:meth:`compute_ordering`)

        - :math:`P_{k}(x) = \\prod_{i=1}^d P_{k_i}^{(a_i, b_i)}(x_i)` is the product of orthonormal Jacobi polynomials

            .. math::

                \\int_{-1}^{1}
                    P_{k}^{(a_i,b_i)}(u) P_{\\ell}^{(a_i,b_i)}(u)
                    (1-u)^{a_i} (1+u)^{b_i} d u
                = \\delta_{k\\ell}

            so that :math:`(P_{k})` are orthonormal w.r.t :math:`\\mu(dx)`

        - :math:`\\Phi(x) = \\left(P_{\\mathfrak{b}^{-1}(0)}(x), \\dots, P_{\\mathfrak{b}^{-1}(N-1)}(x) \\right)^{\\top}`

    :param N:
        Number of points :math:`N \\geq 1`
    :type N:
        int

    :param jacobi_params:
        Jacobi parameters :math:`[(a_i, b_i)]_{i=1}^d`
        The number of rows :math:`d` prescribes the ambient dimension of the points i.e. :math:`x_{1}, \\dots, x_{N} \\in [-1, 1]^d`.
        - when :math:`d=1`, :math:`a_1, b_1 > -1`
        - when :math:`d \\geq 2`, :math:`|a_i|, |b_i| \\leq \\frac{1}{2}`
    :type jacobi_params:
        array_like

    .. seealso::

        - :ref:`multivariate_jacobi_ope`
        - when :math:`d=1`, the :ref:`univariate Jacobi ensemble <jacobi_banded_matrix_model>` is sampled by computing the eigenvalues of a properly randomized :ref:`tridiagonal matrix <jacobi_banded_matrix_model>` of :cite:`KiNe04`
        - :cite:`BaHa16` initiated the use of the multivariate Jacobi ensemble for Monte Carlo integration. In particular, they proved CLT with variance decay of order :math:`N^{-(1+1/d)}` which is faster that the :math:`N^{-1}` rate of vanilla Monte Carlo where the points are drawn i.i.d. from the base measure.
    """

    def __init__(self, N, jacobi_params):

        self.N, self.jacobi_params, self.dim =\
            self._check_params(N, jacobi_params)

        self.ordering = compute_ordering(self.N, self.dim)

        self.deg_max, self.degrees_1D_polynomials =\
            compute_degrees_1D_polynomials(np.max(self.ordering, axis=0))

        self.norms_1D_polynomials =\
            compute_norms_1D_polynomials(self.jacobi_params, self.deg_max)

        self.square_norms_multiD_polynomials =\
            np.prod((self.norms_1D_polynomials**2)[self.ordering,
                                                   range(self.dim)],
                    axis=1)

        self.mass_of_mu = self.square_norms_multiD_polynomials[0]

        self.rejection_bounds =\
            compute_rejection_bounds(self.jacobi_params,
                                     self.ordering,
                                     log_scale=True)

    def _check_params(self, N, jacobi_params):
        """ Check that:

        - The number of points :math:`N \\geq 1`
        - Jacobi parameters
            - when :math:`d=1` we must have :math:`a_1, b_1 > -1`
            - when :math:`d \\geq 2` we must have :math:`|a_i|, |b_i| \\leq \\frac{1}{2}`.
        """
        if type(N) is not int or N < 1:
            raise ValueError('Number of points N={} < 1'.format(N))

        dim = jacobi_params.size // 2

        if dim == 1 and not np.all(jacobi_params > -1):
            raise ValueError('d=1, Jacobi parameters must be > -1')
        elif dim >= 2 and not np.all(np.abs(jacobi_params) <= 0.5):
            raise ValueError('d={}, Jacobi parameters be in [-0.5, 0.5]^d'.format(dim))

        return N, jacobi_params, dim

    def eval_w(self, X):
        """Evaluate :math:`w(x) = \\prod_{i=1}^{d} (1-x_i)^{a_i} (1+x_i)^{b_i}` which corresponds to the density of the base measure :math:`\\mu(dx) = w(x) dx`

        :param X:
            :math:`M\\times d` array of :math:`M` points :math:`\\in [-1, 1]^d`
        :type X:
            array_like

        :return:
            :math:`w(x) = \\prod_{i=1}^{d} (1-x_i)^{a_i} (1+x_i)^{b_i}`
        :rtype:
            array_like
        """
        a, b = self.jacobi_params.T

        return np.prod((1.0 - X)**a * (1.0 + X)**b, axis=-1)

    def eval_multiD_polynomials(self, X):
        """Evaluate

        .. math::

            \\mathbf{\\Phi}(X)
                := \\begin{pmatrix}
                    \\Phi(x_1)^{\\top}\\\\
                    \\vdots\\\\
                    \\Phi(x_M)^{\\top}
                  \\end{pmatrix}

        where :math:`\\Phi(x) = \\left(P_{\\mathfrak{b}^{-1}(0)}(x), \\dots, P_{\\mathfrak{b}^{-1}(N-1)}(x) \\right)^{\\top}` such that
        :math:`K(x, y) = \\Phi(x)^{\\top} \\Phi(y)`.
        Recall that :math:`\\mathfrak{b}` denotes the ordering chosen to order multi-indices :math:`k\\in \\mathbb{N}^d`.

        This is done by evaluating each of the `three-term recurrence relations <https://en.wikipedia.org/wiki/Jacobi_polynomials#Recurrence_relations>`_ satisfied by each univariate orthogonal Jacobi polynomial, using the dedicated `see also SciPy <https://docs.scipy.org/doc/scipy/reference/generated/scipy.special.eval_jacobi.html>`_ :func:`scipy.special.eval_jacobi` satistified by the respective univariate Jacobi polynomials :math:`P_{k_i}^{(a_i, b_i)}(x_i)`.
        Then we use the slicing feature of the Python language to compute :math:`\\Phi(x)=\\left(P_{k}(x) = \\prod_{i=1}^d P_{k_i}^{(a_i, b_i)}(x_i)\\right)_{k=\\mathfrak{b}^{-1}(0), \\dots, \\mathfrak{b}^{-1}(N-1)}^{\\top}`

        :param X:
            :math:`M\\times d` array of :math:`M` points :math:`\\in [-1, 1]^d`
        :type X:
            array_like

        :return:
            :math:`\\mathbf{\\Phi}(X)` - :math:`M\\times N` array
        :rtype:
            array_like

        .. seealso::

            - evaluation of the kernel :py:meth:`~dppy.multivariate_jacobi_ope.MultivariateJacobiOPE.K`
        """
        poly_1D_jacobi = eval_jacobi(self.degrees_1D_polynomials,
                                     self.jacobi_params[:, 0],
                                     self.jacobi_params[:, 1],
                                     np.atleast_2d(X)[:, None])\
                        / self.norms_1D_polynomials

        return np.prod(poly_1D_jacobi[:, self.ordering, range(self.dim)],
                       axis=2)

    def K(self, X, Y=None, eval_pointwise=False):
        """Evalute :math:`\\left(K(x, y)\\right)_{x\\in X, y\\in Y}` if ``eval_pointwise=False`` or :math:`\\left(K(x, y)\\right)_{(x, y)\\in (X, Y)}` otherwise

        .. math::

            K(x, y) = \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                        P_{k}(x) P_{k}(y)
                    = \\phi(x)^{\\top} \\phi(y)

        where

        - :math:`k \\in \\mathbb{N}^d` is a multi-index ordered according to the ordering :math:`\\mathfrak{b}`, :py:meth:`compute_ordering`

        - :math:`P_{k}(x) = \\prod_{i=1}^d P_{k_i}^{(a_i, b_i)}(x_i)` is the product of orthonormal Jacobi polynomials

            .. math::

                \\int_{-1}^{1}
                    P_{k}^{(a_i,b_i)}(u) P_{\\ell}^{(a_i,b_i)}(u)
                    (1-u)^{a_i} (1+u)^{b_i} d u
                = \\delta_{k\\ell}

            so that :math:`(P_{k})` are orthonormal w.r.t :math:`\\mu(dx)`

        - :math:`\\Phi(x) = \\left(P_{\\mathfrak{b}^{-1}(0)}(x), \\dots, P_{\\mathfrak{b}^{-1}(N-1)}(x) \\right)`, see :py:meth:`eval_multiD_polynomials`

        :param X:
            :math:`M\\times d` array of :math:`M` points :math:`\\in [-1, 1]^d`
        :type X:
            array_like

        :param Y:
            :math:`M'\\times d` array of :math:`M'` points :math:`\\in [-1, 1]^d`
        :type Y:
            array_like (default None)

        :param eval_pointwise:
            sets pointwise evaluation of the kernel, if ``True``, :math:`X` and :math:`Y` must have the same shape, see Returns
        :type eval_pointwise:
            bool (default False)

        :return:

            If ``eval_pointwise=False`` (default), evaluate the kernel matrix

            .. math::

                \\left(K(x, y)\\right)_{x\\in X, y\\in Y}

            If ``eval_pointwise=True`` kernel matrix
            Pointwise evaluation of :math:`K` as depicted in the following pseudo code output

            - if ``Y`` is ``None``

                - :math:`\\left(K(x, y)\\right)_{x\\in X, y\\in X}` if ``eval_pointwise=False``
                - :math:`\\left(K(x, x)\\right)_{x\\in X}` if ``eval_pointwise=True``

            - otherwise

                - :math:`\\left(K(x, y)\\right)_{x\\in X, y\\in Y}` if ``eval_pointwise=False``
                - :math:`\\left(K(x, y)\\right)_{(x, y)\\in (X, Y)}` if ``eval_pointwise=True`` (in this case X and Y should have the same shape)
        :rtype:
            array_like

        .. seealso::

            :py:meth:`eval_multiD_polynomials`
        """

        X = np.atleast_2d(X)

        if Y is None or Y is X:
            phi_X = self.eval_multiD_polynomials(X)
            if eval_pointwise:
                return inner1d(phi_X, phi_X, axis=1)
            else:
                return phi_X.dot(phi_X.T)
        else:
            len_X = len(X)
            phi_XY = self.eval_multiD_polynomials(np.vstack((X, Y)))
            if eval_pointwise:
                return inner1d(phi_XY[:len_X], phi_XY[len_X:], axis=1)
            else:
                return phi_XY[:len_X].dot(phi_XY[len_X:].T)

    def sample_chain_rule_proposal(self, nb_trials_max=10000,
                                   random_state=None):
        """Use a rejection sampling mechanism to sample

        .. math::

            \\frac{1}{N} K(x, x) w(x) dx
            = \\frac{1}{N}
                \\sum_{\\mathfrak{b}(k)=0}^{N-1}
                \\left( \\frac{P_k(x)}{\\left\\| P_k \\right\\|} \\right)^2
                w(x)

        with proposal distribution

        .. math::

            w_{eq}(x) d x
                = \\prod_{i=1}^{d}
                    \\frac{1}{\\pi\\sqrt{1-(x_i)^2}}
                    d x_i

        Since the target density is a mixture, we can sample from it by

        1. Select a multi-index :math:`k` uniformly at random in :math:`\\left\\{ \\mathfrak{b}^{-1}(0), \\dots, \\mathfrak{b}^{-1}(N-1) \\right\\}`
        2. Sample from :math:`\\left( \\frac{P_k(x)}{\\left\\| P_k \\right\\|} \\right)^2 w(x) dx` with proposal :math:`w_{eq}(x) d x`.

            The acceptance ratio writes

            .. math::

                \\frac{
                    \\left( \\frac{P_k(x)}{\\left\\| P_k \\right\\|} \\right)^2
                        w(x)}
                    {w_{eq}(x)}
                = \\prod_{i=1}^{d}
                    \\pi
                    \\left(
                        \\frac{P_{k_i}^{(a_i, b_i)}(x)}
                               {\\left\\| P_{k_i}^{(a_i, b_i)} \\right\\|}
                    \\right)^2
                    (1-x_i)^{a_i+\\frac{1}{2}}
                    (1+x_i)^{b_i+\\frac{1}{2}}
                \\leq C_{k}

            which can be bounded using the result of :cite:`Gau09` on Jacobi polynomials.

            .. note::

                Each of the rejection constant :math:`C_{k}` is computed at initialization of the :py:class:`MultivariateJacobiOPE` object using :py:meth:`compute_rejection_bounds`

        :return:
            A sample :math:`x\\in[-1,1]^d` with probability distribution :math:`\\frac{1}{N} K(x,x) w(x)`
        :rtype:
            array_like

        .. seealso::

            - :py:meth:`compute_rejection_bounds`
            - :py:meth:`sample`
        """
        rng = check_random_state(random_state)

        a, b = self.jacobi_params.T
        a_05, b_05 = a + 0.5, b + 0.5
        d = self.dim

        ind = rng.randint(self.N)
        k = self.ordering[ind]
        Pk_square_norm = self.square_norms_multiD_polynomials[ind]
        # norm_Pk = self.poly_multiD_norm[ind]
        rejection_bound = self.rejection_bounds[ind]

        for trial in range(nb_trials_max):

            # Propose x ~ w_eq(x) = \prod_{i=1}^{d} 1/pi 1/sqrt(1-(x_i)^2)
            # rng.beta is defined as beta(a, b) = x^(a-1) (1-x)^(b-1)
            x = 1.0 - 2.0 * rng.beta(0.5, 0.5, size=self.dim)

            # Compute (P_k(x)/||P_k||)^2
            Pk2_x = np.prod(eval_jacobi(k, a, b, x))**2 / Pk_square_norm
            # Pk2_x = (np.prod(eval_jacobi(k, a, b, x)) / norm_Pk)**2

            # Compute w(x) / w_eq(x)
            w_over_w_eq =\
                np.pi**d * np.prod((1.0 - x)**a_05 * (1.0 + x)**b_05)

            if rng.rand() * rejection_bound < Pk2_x * w_over_w_eq:
                break
        else:
            print('marginal distribution 1/N K(x,x), rejection fails after {} proposals'.format(trial))

        return x

    def sample(self, nb_trials_max=10000, random_state=None, tridiag_1D=True):
        """Use the chain rule :cite:`HKPV06` (Algorithm 18) to sample :math:`\\left(x_{1}, \\dots, x_{N} \\right)` with density

        .. math::

            & \\frac{1}{N!}
                \\left(K(x_n,x_p)\\right)_{n,p=1}^{N}
                \\prod_{n=1}^{N} w(x_n)\\\\
            &= \\frac{1}{N} K(x_1,x_1) w(x_1)
            \\prod_{n=2}^{N}
                \\frac{
                    K(x_n,x_n)
                    - K(x_n,x_{1:n-1})
                    \\left[\\left(K(x_k,x_l)\\right)_{k,l=1}^{n-1}\\right]^{-1}
                    K(x_{1:n-1},x_n)
                    }{N-(n-1)}
                    w(x_n)\\\\
            &= \\frac{\\| \\Phi(x) \\|^2}{N} \\omega(x_1) d x_1
            \\prod_{n=2}^{N}
                \\frac{\\operatorname{distance}^2(\\Phi(x_n), \\operatorname{span}\\{\\Phi(x_p)\\}_{p=1}^{n-1})}
                {N-(n-1)}
            \\omega(x_n) d x_n

        The order in which the points were sampled can be forgotten to obtain a valid sample of the corresponding DPP

        - :math:`x_1 \\sim \\frac{1}{N} K(x,x) w(x)` using :py:meth:`sample_chain_rule_proposal`

        - :math:`x_n | Y = \\left\\{ x_{1}, \\dots, x_{n-1} \\right\\}`, is sampled using rejection sampling with proposal density :math:`\\frac{1}{N} K(x,x) w(x)` and rejection bound \\frac{N}{N-(n-1)}

            .. math::

                \\frac{1}{N-(n-1)} [K(x,x) - K(x, Y) K_Y^{-1} K(Y, x)] w(x)
                \\leq \\frac{N}{N-(n-1)} \\frac{1}{N} K(x,x) w(x)

        .. note::

            Using the gram structure :math:`K(x, y) = \\Phi(x)^{\\top} \\Phi(y)` the numerator of the successive conditionals reads

            .. math::

                K(x, x) - K(x, Y) K(Y, Y)^{-1} K(Y, x)
                &= \\operatorname{distance}^2(\\Phi(x_n), \\operatorname{span}\\{\\Phi(x_p)\\}_{p=1}^{n-1})\\\\
                &= \\left\\| (I - \\Pi_{\\operatorname{span}\\{\\Phi(x_p)\\}_{p=1}^{n-1}} \\phi(x)\\right\\|^2

            which can be computed simply in a vectorized way.
            The overall procedure is akin to a sequential Gram-Schmidt orthogonalization of :math:`\\Phi(x_{1}), \\dots, \\Phi(x_{N})`.

        .. seealso::

            - :ref:`continuous_dpps_exact_sampling_projection_dpp_chain_rule`
            - :py:meth:`sample_chain_rule_proposal`
        """

        rng = check_random_state(random_state)

        if self.dim == 1 and tridiag_1D:
            sample = tridiagonal_model(a=self.jacobi_params[0, 0] + 1,
                                       b=self.jacobi_params[0, 1] + 1,
                                       size=self.N,
                                       random_state=rng)[:, None]
            return 1.0 - 2.0 * sample

        sample = np.zeros((self.N, self.dim))
        phi = np.zeros((self.N, self.N))

        for n in range(self.N):

            for trial in range(nb_trials_max):

                # Propose a point ~ 1/N K(x,x) w(x)
                sample[n] = self.sample_chain_rule_proposal(random_state=rng)

                # Schur complement (numerator of x_n | Y = x_1:n-1)
                #   = K(x, x) - K(x, Y) K(Y, Y)^-1 K(Y, x)
                #   = ||(I - Proj{phi(Y)}) phi(x)||^2
                phi[n] = self.eval_multiD_polynomials(sample[n])
                K_xx = phi[n].dot(phi[n])  # self.K(sample[n], sample[n])
                phi[n] -= phi[:n].dot(phi[n]).dot(phi[:n])
                schur = phi[n].dot(phi[n])

                # accept: x_n = x, or reject
                if rng.rand() < schur / K_xx:
                    # normalize phi(x_n) / ||phi(x_n)||
                    phi[n] /= np.sqrt(schur)
                    break
            else:
                print('conditional x_{} | x_1,...,x_{}, rejection fails after {} proposals'.format(n + 1, n, trial))

        return sample

    def plot(self, sample, weighted=''):

        if self.dim >= 3:
            raise NotImplementedError('Visualizations in d>=3 not implemented')

        tols = 5e-2 * np.ones_like(self.jacobi_params)
        tols = np.zeros_like(self.jacobi_params)
        tols[1, 0] = 8e-2

        weights = np.ones(len(sample))

        if weighted == 'BH':
            # w_n = 1 / K(x_n, x_n)
            weights = 1. / self.K(sample, eval_pointwise=True)

        elif weighted == 'EZ':
            Phi_X = self.eval_multiD_polynomials(sample)

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

        if self.dim == 1:

            fig, ax_main = plt.subplots(figsize=(6, 4))

            ax_main.tick_params(axis='both', which='major', labelsize=18)
            ax_main.set_xticks(ticks_pos)
            ax_main.set_xticklabels(ticks_labs)

            ax_main.spines['right'].set_visible(False)
            ax_main.spines['top'].set_visible(False)

            ax_main.scatter(sample[:, 0],
                            np.zeros_like(sample[:, 0]),
                            s=weights)

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

            ax_main.legend(fontsize=15,
                           loc='center',
                           bbox_to_anchor=(0.5, -0.15 if weighted else -0.17),
                           labelspacing=0.1,
                           frameon=False)

        elif self.dim == 2:

            # Create Fig and gridspec
            fig = plt.figure(figsize=(6, 6))
            grid = plt.GridSpec(6, 6, hspace=0., wspace=0.)

            ax_main = fig.add_subplot(grid[1:, :-1],
                                      xticks=ticks_pos, xticklabels=ticks_labs,
                                      yticks=ticks_pos, yticklabels=ticks_labs)

            ax_main.tick_params(axis='both', which='major', labelsize=18)

            if weighted == 'EZ':
                weights *= 100
                w_geq_0 = weights >= 0
                ax_main.scatter(sample[w_geq_0, 0],
                                sample[w_geq_0, 1],
                                s=weights[w_geq_0], alpha=0.7)

                ax_main.scatter(sample[~w_geq_0, 0],
                                sample[~w_geq_0, 1],
                                s=-weights[~w_geq_0], alpha=0.7)
            else:
                weights *= 20
                ax_main.scatter(sample[:, 0],
                                sample[:, 1],
                                s=weights, alpha=0.8)

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
                        weights=np.abs(weights),
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
                          weights=np.abs(weights),
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
                           fontsize=15,
                           loc='center',
                           bbox_to_anchor=(0.5, -0.15 if weighted else -0.18),
                           labelspacing=0.1,
                           frameon=False)


def compute_ordering(N, d):
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
                filter(lambda x: m in x,
                       itt.product(range(m + 1), repeat=d))
                for m in range(layer_max + 1))

    return list(ordering)[:N]


def compute_norms_1D_polynomials(jacobi_params, deg_max):
    """ Compute the square norms :math:`\\|P_{k}^{(a_i,b_i)}\\|^2` of each (univariate) orthogoanl Jacobi polynomial for :math:`k=0` to ``deg_max`` and :math:`a_i, b_i =` ``jacobi_params[i, :]``
    Recall that the Jacobi polynomials :math:`\\left( P_{k}^{(a_i,b_i)} \\right)` are `orthogonal <http://en.wikipedia.org/wiki/Jacobi_polynomials#Orthogonality>`_ w.r.t. :math:`(1-u)^{a_i} (1+u)^{b_i} du`.

    .. math::

        \\|P_{k}^{(a_i,b_i)}\\|^2
            &= \\int_{-1}^{1}
                \\left( P_{k}^{(a_i,b_i)}(u) \\right)^2
                (1-u)^{a_i} (1+u)^{b_i} d u\\\\
            &= \\frac{2^{a_i+b_i+1}}
                    {2k+a_i+b_i+1}
              \\frac{\\Gamma(k+a_i+1)\\Gamma(k+b_i+1)}
                    {\\Gamma(k+a_i+b_i+1)n!}

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
        - :py:meth:`compute_ordering`
    """

    # Initialize
    # - [square_norms]_ij = ||P_i^{a_j, b_j}||^2
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

        square_norms[0, non_arcsine] = 2**(a + b + 1) * beta(a + 1, b + 1)

        square_norms[1:, non_arcsine] = np.exp((a + b + 1) * np.log(2)
                                               + gammaln(n + 1 + a)
                                               + gammaln(n + 1 + b)
                                               - gammaln(n + 1)
                                               - np.log(2 * n + 1 + a + b)
                                               - gammaln(n + 1 + a + b))

    return np.sqrt(square_norms)


def compute_rejection_bounds(jacobi_params, ordering, log_scale=True):
    """ Compute the rejection constants for the acceptance/rejection mechanism used in :py:meth:`sample_chain_rule_proposal` to sample

    .. math::

        \\frac{1}{N} K(x, x) w(x) dx
        = \\frac{1}{N}
        \\sum_{\\mathfrak{b}(k)=0}^{N-1}
        \\left( \\frac{P_k(x)}{\\left\\| P_k \\right\\|} \\right)^2
        w(x)

    with proposal distribution

    .. math::

        w_{eq}(x) d x
        = \\prod_{i=1}^{d} \\frac{1}{\\pi\\sqrt{1-(x_i)^2}} d x_i

    To get a sample:

    1. Draw a multi-index :math:`k` uniformly at random in :math:`\\left\\{ \\mathfrak{b}^{-1}(0), \\dots, \\mathfrak{b}^{-1}(N-1) \\right\\}`
    2. Sample from :math:`P_k(x)^2 w(x) dx` with proposal :math:`w_{eq}(x) d x`.

        The acceptance ratio writes

        .. math::

            \\frac{\\left( \\frac{P_k(x)}{\\left\\| P_k \\right\\|} \\right)^2
                   w(x)}
                  {w_{eq}(x)}
            = \\prod_{i=1}^{d}
                \\pi
                \\left(
                    \\frac{P_{k_i}^{(a_i, b_i)}(x)}
                          {\\left\\| P_{k_i}^{(a_i, b_i)} \\right\\|}
                \\right)^2
                (1-x_i)^{a_i+\\frac{1}{2}}
                (1+x_i)^{b_i+\\frac{1}{2}}
            \\leq C_k

        - For :math:`k_i>0` we use a result on Jacobi polynomials given by, e.g., :cite:`Gau09`, for :math:`\\quad|a|,|b| \\leq \\frac{1}{2}`

        .. math::

            &
                \\pi
                (1-u)^{a+\\frac{1}{2}}
                (1+u)^{b+\\frac{1}{2}}
                \\left(
                    \\frac{P_{n}^{(a, b)}(u)}
                          {\\left\\| P_{n}^{(a, b)} \\right\\|}
                \\right)^2\\\\
            &\\leq
                \\frac{2}
                      {n!(n+(a+b+1) / 2)^{2 \\max(a,b)}}
                \\frac{\\Gamma(n+a+b+1)
                        \\Gamma(n+\\max(a,b)+1)}
                      {\\Gamma(n+\\min(a,b)+1)}

        - For :math:`k_i=0`, we use less involved properties of the `Jacobi polynomials <https://en.wikipedia.org/wiki/Jacobi_polynomials>`_:

            - :math:`P_{0}^{(a, b)} = 1`
            - :math:`\\|P_{0}^{(a, b)}\\|^2 = 2^{a+b+1} \\operatorname{B}(a+1,b+1)`
            - :math:`m = \\frac{b-a}{a+b+1}` is the mode of :math:`(1-u)^{a+\\frac{1}{2}} (1+u)^{b+\\frac{1}{2}}` (valid since :math:`a+\\frac{1}{2}, b+\\frac{1}{2} > 0`)

            So that,

            .. math::

                    \\pi
                    (1-u)^{a+\\frac{1}{2}}
                    (1+u)^{b+\\frac{1}{2}}
                    \\left(\\frac{P_{0}^{(a, b)}(u)}
                           {\\|P_{0}^{(a, b)}\\|}\\right)^{2}
                &=
                    \\frac
                    {\\pi
                     (1-u)^{a+\\frac{1}{2}}
                     (1+u)^{b+\\frac{1}{2}}}
                    {\\|P_{0}^{(a, b)}\\|^2} \\\\
                &\\leq
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
        Ordering of the multi-indices :math:`\\in\\mathbb{N}^d` defining the order between the multivariate monomials (see also :py:meth:`compute_ordering`)

        - the number of rows corresponds to the number :math:`N` of monomials considered.
        - the number of columns :math:`=d`

    :type ordering:
        array_like

    :param log_scale:
        If True, the rejection bound is computed using the logarithmic versions ``betaln``, ``gammaln`` of ``beta`` and ``gamma`` functions to avoid overflows

    :type log_scale:
        bool

    :return:
        The rejection bounds :math:`C_{k}` for :math:`k = \\mathfrak{b}^{-1}(0), \\dots, \\mathfrak{b}^{-1}(N-1)`

    :rtype:
        array_like

    .. seealso::

        - :cite:`Gau09` for the domination when :math:`k_i > 0`
        - :py:meth:`compute_poly1D_norms`
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


def compute_degrees_1D_polynomials(max_degrees):
    """ deg[i, j] = i if i <= max_degrees[j] else 0
    """

    max_deg, dim = max(max_degrees), len(max_degrees)
    degrees = np.tile(np.arange(max_deg + 1)[:, None], (1, dim))
    degrees[degrees > max_degrees] = 0

    return max_deg, degrees
