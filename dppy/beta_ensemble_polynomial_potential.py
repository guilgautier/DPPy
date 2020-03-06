# coding: utf8

import numpy as np
from numpy.polynomial.chebyshev import poly2cheb  # cf rescale_largest_eig_val

import scipy.linalg as la

import matplotlib.pyplot as plt

from dppy.beta_ensemble_polynomial_potential_core import (polynomial_in_negative_log_conditional_a_coef as P_a_cond,
    polynomial_in_negative_log_conditional_b_coef as P_b_cond,
    sampler_exact_convex_quartic,
    sampler_mala,
    equilibrium_x2m,
    equilibrium_x2_x4,
    equilibrium_ClItKr10,
    check_random_state)


class BetaEnsemblePolynomialPotential(object):
    """docstring for BetaEnsemblePolynomialPotential"""
    def __init__(self, beta, potential_V, name=None):

        self.beta = beta
        self.V = potential_V

        if not (self.beta > 0):
            raise ValueError('beta = {} <= 0'.format(beta))

        if self.V.order > 7:

            str_ = ['Polynomial potentials V are allowed up to degree 6',
                    'Given\n',
                    ' '.join(['g_{}={}'.format(n, g_n)
                              for n, g_n in enumerate(self.V.coef[::-1])])]
            raise ValueError(' '.join(str_))

        if self.V[5]:
            raise ValueError('Potentials V = ... + g_5 x^5 + ... not supported, given g_5={}'.format(self.V[5]))

        if self.V.order % 2:
            raise ValueError('V has odd degree deg(V)={}'.format(self.V.order))

        if self.V[self.V.order] < 0:
            raise ValueError('V = g_{} x^{} + ... with g_{} < 0'.format(*[self.V.order] * 3))

        if self.V[0] != 0:
            print('Weird thing to introduce a constant V = ... + g_0={}. Reset g_0 = 0'.format(self.V[0]))
            self.V[0] = 0

        self._V_ClItKr10 = np.poly1d([1 / 20, -4 / 15, 1 / 5, 8 / 5, 0])

        self.equilibrium_density, self.support = self.__compute_equilibrium()

        # Sampling
        self.N = None
        self.nb_gibbs_passes = None

    def __str__(self):
        return '\n'.join(['beta={}'.format(self.beta),
                          'V(x) =\n{}'.format(self.V.__str__())])

    def __compute_equilibrium(self):
        """ Update attribute equilibrium_density and support if available
        """

        equi_dens, supp = None, None

        deg_V = self.V.order
        deg_V_2, deg_V_odd = divmod(deg_V, 2)

        set_non_zero_coefs = set(np.nonzero(self.V.coef[::-1])[0])

        if not deg_V_odd:
            if self.V == self._V_ClItKr10:
                equi_dens, supp = equilibrium_ClItKr10()
            elif set_non_zero_coefs == {deg_V}:
                equi_dens, supp = equilibrium_x2m(deg_V_2,
                                                  deg_V * self.V[deg_V])
            elif set_non_zero_coefs == {2, 4}:
                equi_dens, supp = equilibrium_x2_x4(2 * self.V[2],
                                                    4 * self.V[4])

        return equi_dens, supp

    def sample_mcmc(self,
                    N=10,
                    nb_gibbs_passes=10,
                    sample_exact_cond=False,
                    nb_mala_steps=100,
                    return_chain_of_eig_vals=False,
                    return_chain_of_lambda_max=False,
                    random_state=None):
        """ Gibbs sampler on Jacobi matrices to sample approximately from the corresponding :math:`\\beta`-ensemble.

        :param N:
            Number of points/size of the :math:`\\beta`-ensemble
        :type N:
            int

        :param nb_gibbs_passes:
            Number of passes/sweeps over the variables using the Gibbs sampler
        :type nb_gibbs_passes:
            int

        :param sample_exact_cond:
            Flag to force (``True``) exact sampling from the conditionals when it is possible.
            Otherwise run MALA for ``nb_mala_steps` to sample from the conditionals.
        :type sample_exact_cond:
            bool (default 100)

        :param nb_mala_steps:
            Number of steps of Metropolis Ajusted Langevin Algorithm (MALA) to perform when the conditionals are sampled approximately
        :type nb_mala_steps:
            int, default 100

        :param return_chain_of_eig_vals:
            Flag to return the chain of eigenvalues associated to the chain of Jacobi matrices.
            If ``True`` the whole chain of eigenvalues is returned
            If ``False`` only the last sequence of eigenvalues is returned
        :type return_chain_of_eig_vals:
            bool (default False)

        :param return_chain_of_lambda:
            Flag to return the chain of the **largest** eigenvalues associated to the chain of Jacobi matrices.
            If ``True`` the whole chain of the **largest** eigenvalues is returned
            If ``False`` only the **largest** eigenvalue of the last Jacobi matrix is returned
        :type return_chain_of_eig_vals:
            bool (default False)
        """

        rng = check_random_state(random_state)

        if sample_exact_cond:
            if self.V[3]:
                raise ValueError('Sampling exactly the conditionals a_i |... from V = ... + x^3 + ... is not supported, given g_3={}. Conditionals are not log-concave, cannot use Dev12 sampler'.format(self.V[3]))

            if self.V.order >= 5:
                raise ValueError('Sampling exactly the conditionals a_i |... from V = ... + x^5 + ... is not supported, deg(V)={}>=5. Conditionals are not log-concave, cannot use Dev12 sampler'.format(self.V.order))

            even_coefs_V = self.V.coef[::-1][2::2]
            if not all(even_coefs_V >= 0):
                raise ValueError('\n'.join(
                    ['even coefs of V are not all >=0',
                     ', '.join(['g_{}={}'.format(2 * (n + 1), g_2n)
                                for n, g_2n in enumerate(even_coefs_V)]),
                     'Conditionals are not log-concave, cannot use Dev12 sampler',
                     'You may retry swithching `sample_exact_cond` to False']))

        self.N = N
        self.nb_gibbs_passes = nb_gibbs_passes

        a, b = np.zeros((2, N + 3))

        if return_chain_of_eig_vals:
            eig_vals = np.zeros((N, nb_gibbs_passes))
        elif return_chain_of_lambda_max:
            lambda_max = np.zeros(nb_gibbs_passes)

        for p in range(nb_gibbs_passes):
            if (p + 1) % 50 == 0:
                print(p + 1)

            for i in range(1, N + 1):

                # a_i | ... propto exp - P_a_i
                P_a_i = 0.5 * self.beta * N * P_a_cond(i, a, b, self.V)
                if sample_exact_cond:
                    a[i], _ = sampler_exact_convex_quartic(
                                P=P_a_i,
                                random_state=rng)
                else:
                    a[i] = sampler_mala(a[i],
                                        V=P_a_i,
                                        sigma=0.01,
                                        nb_steps=nb_mala_steps,
                                        random_state=rng)

                # b_i | ... propto x^(shape-1) * exp - P_b_i
                if i < N:
                    P_b_i = 0.5 * self.beta * N * P_b_cond(i, a, b, self.V)
                    b[i], _ = sampler_exact_convex_quartic(
                                P=P_b_i,
                                shape=0.5 * self.beta * (N - i),
                                random_state=rng)

            if return_chain_of_eig_vals:
                eig_vals[:, p] = la.eigvalsh_tridiagonal(a[1:N + 1],
                                                         np.sqrt(b[1:N]))
            elif return_chain_of_lambda_max:
                lambda_max[p] = la.eigvalsh_tridiagonal(
                                    a[1:N + 1],
                                    np.sqrt(b[1:N]),
                                    select='i',
                                    select_range=(N - 1, N - 1))[0]

        if return_chain_of_eig_vals:
            return eig_vals
        if return_chain_of_lambda_max:
            return lambda_max

        return la.eigvalsh_tridiagonal(a[1:N + 1], np.sqrt(b[1:N]))

    def hist(self, sampl, save_file_name=False):
        """ Display the histogram of a ``sampl`` from the corresponding :math:`\\beta`-ensemble and the corresponding equilibrium distribution when available

        :param sampl:
            One or multiple samples from the corresponding :math:`\\beta`-ensemble.
            **In any case ``sampl`` is flattened** as if the samples were concatenated
        :type sampl:
            array_like

        :param save_file_name:
            File name, e.g. ``figure.pdf``, to save the plot
        :type save_file_name:
            str

        .. seealso::

            :py:func:`__compute_equilibrium`
        """

        fig, ax = plt.subplots(1, 1)

        # Title
        # V_x = ' '.join(['V(x) =',
        #                 ' + '.join([r'$g_{} x^{}$'.format(n, n)
        #                             for n, g_n in enumerate(self.V.coef[::-1],
        #                                                     start=0)
        #                             if g_n])])

        # with_coefs = ', '.join([r'$g_{}={:0.2f}$'.format(n, g_n)
        #                         for n, g_n in enumerate(self.V.coef[::-1])
        #                         if g_n])

        # beta_N_passes = r'$\beta={}, N={}$ #Gibbs_passes={}'.format(
        #                 self.beta, self.N, self.nb_gibbs_passes)
        # plt.title('\n'.join([V_x, with_coefs, beta_N_passes]))

        # histogram
        ax.hist(np.ravel(sampl),
                density=True,
                histtype='step',
                lw=3,
                bins=30,
                label='histogram')

        if self.equilibrium_density is not None and self.support is not None:
            # equilibrium_measure
            x = np.linspace(1.1 * self.support[0],
                            1.1 * self.support[1],
                            300)
            ax.plot(x, self.equilibrium_density(x),
                    label=r'$\mu_{eq}$', lw=3, c='k')

        # start, end = ax.get_xlim()
        # ax.xaxis.set_ticks(np.arange(-1.5, 2.1, 1.5))
        ax.xaxis.set_ticks(np.arange(-2, 2.1, 1))

        plt.legend(loc='best',
                   fontsize='x-large',
                   frameon=False,
                   handlelength=1)
        ax.tick_params(axis='both', which='major', labelsize=15)
        plt.tight_layout()

        if save_file_name:
            plt.savefig(save_file_name)

    def rescale_largest_eig_val(self, lambda_max):
        """ Rescale the largest eigenvalue to see Tracy-Widom fluctuations

        .. math::

            N^{\\frac{2}{3}} c_v (\\lambda_{\\max} - b_v)

        where

        .. math::

            c_V = (b_v - a_v)^{-\\frac{1}{3}} \\left(\\sum_{n=1}^{\\infty} k {V'}_k \\right)^{\\frac{2}{3}}

        with :math:`{V'}_k` being the Chebychev coefficients of

        .. math::

            V'(\\frac{a_v + b_v}{2} + \\frac{b_v - a_v}{2} X)

        .. seealso::

            - Section 3.2 https://arxiv.org/pdf/1210.2199.pdf
            - :cite:`OlNaTr14` p.5 Equation 2.3 `https://arxiv.org/pdf/1404.0071.pdf <https://arxiv.org/pdf/1404.0071.pdf>`_
            - `poly2cheb <https://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.chebyshev.poly2cheb.html?>`_
        """

        a_v, b_v = self.support
        shift = np.poly1d([0.5 * (b_v - a_v), 0.5 * (b_v + a_v)])

        dV_shift = self.V.deriv(m=1)(shift)

        dV_shift_cheb = poly2cheb(dV_shift.coeffs[::-1])

        sum_k_dV_k = sum(k * dV_k
                         for k, dV_k in enumerate(dV_shift_cheb[1:], start=1))

        c_v = np.cbrt(sum_k_dV_k**2 / (b_v - a_v))

        return self.N**(2 / 3) * c_v * (lambda_max - b_v)
