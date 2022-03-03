import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import dppy.random_matrices as rm
from dppy.beta_ensembles.abstract_beta_ensemble import AbstractBetaEnsemble
from dppy.utils import check_random_state


class JacobiBetaEnsemble(AbstractBetaEnsemble):
    """Jacobi Ensemble object

    .. seealso::

        - :ref:`Full matrix model <jacobi_full_matrix_model>` associated to the Jacobi ensemble
        - :ref:`Tridiagonal matrix model <jacobi_banded_matrix_model>` associated to the Jacobi ensemble
    """

    def __init__(self, beta=2):

        super().__init__(beta=beta)

        params = {"a": 1.0, "b": 1.0, "N": 10, "M1": None, "M2": None}
        self.params.update(params)

    def sample_full_model(self, N=100, M1=150, M2=200, random_state=None):
        """Sample from :ref:`full matrix model <Jacobi_full_matrix_model>` associated to the Jacobi ensemble. Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\operatorname{Beta}(a,b)` reference measure

        :param N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized.
            First dimension of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type N:
            int, default :math:`100`

        :param M1:
            Second dimension :math:`M_1` of the first matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type M1:
            int, default :math:`150`

        :param M2:
            Second dimension :math:`M_2` of the second matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type M2:
            int, default :math:`200`

        .. note::

            The reference measure associated with the :ref:`full matrix model <jacobi_full_matrix_model>` is

            .. math::

                \\operatorname{Beta}\\left(\\frac{\\beta}{2}(M_1-N+1), \\frac{\\beta}{2}(M_2-N+1)\\right)

            For this reason, in the :py:attr:`sampling_params` attribute, the values of the parameters are set to ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``N``, ``M2`` and ``M2`` parameters.

        .. seealso::

            - :ref:`Full matrix model <Jacobi_full_matrix_model>` associated to the Jacobi ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "full"

        if (M1 >= N) and (M2 >= N):
            # all([var >= N for var in [M1, M2]]
            a = 0.5 * self.beta * (M1 - N + 1)
            b = 0.5 * self.beta * (M2 - N + 1)

        else:
            err_print = (
                "Must have M1, M2 >= N.",
                "Given: M1={}, M2={} and N={}".format(M1, M2, N),
            )
            raise ValueError(" ".join(err_print))

        params = {
            "a": a,
            "b": b,
            "N": N,
            "M1": M1,
            "M2": M2,
        }
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample i.i.d. Beta(a,b) if M1,2 were used a,b = beta/2 (M_1,2 - N + 1) = 0 => ERROR
            sampl = rng.beta(
                a=self.params["a"], b=self.params["b"], size=self.params["N"]
            )
        else:
            sampl = rm.jacobi_sampler_full(
                M_1=self.params["M1"],
                M_2=self.params["M2"],
                N=self.params["N"],
                beta=self.beta,
                random_state=rng,
            )

        self.list_of_samples.append(sampl)
        return sampl

    def sample_banded_model(
        self, a=1.0, b=2.0, N=10, M1=None, M2=None, random_state=None
    ):
        """Sample from :ref:`tridiagonal matrix model <Jacobi_banded_matrix_model>` associated to the Jacobi ensemble. Available for :py:attr:`beta` :math:`>0` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\operatorname{Beta}(a,b)` reference measure

        :param shape:
            Shape parameter :math:`k` of :math:`\\Gamma(k, \\theta)` reference measure
        :type shape:
            float, default :math:`1`

        :param scale:
            Scale parameter :math:`\\theta` of :math:`\\Gamma(k, \\theta)` reference measure
        :type scale:
            float, default :math:`2.0`

        :param N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized.
            Equivalent to the first dimension :math:`N` of the matrices used in the :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type N:
            int, default :math:`10`

        :param M1:
            Equivalent to the second dimension :math:`M_1` of the first matrix used in the :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type M1:
            int

        :param M2:
            Equivalent to the second dimension :math:`M_2` of the second matrix used in the :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type M2:
            int

        .. note::

            The reference measure associated with the :ref:`full matrix model <jacobi_full_matrix_model>` is :

            .. math::

                \\operatorname{Beta}\\left(\\frac{\\beta}{2}(M_1-N+1), \\frac{\\beta}{2}(M_2-N+1)\\right)

            For this reason, in the :py:attr:`sampling_params` attribute, the values of the parameters are set to ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``N``, ``M2`` and ``M2`` parameters.

        - If ``M1`` and ``M2`` are not provided:

            In the :py:attr:`sampling_params` attribute, ``M1,2`` are set to
            ``M1``:math:`= \\frac{2a}{\\beta} + N - 1` and ``M2``:math:`= \\frac{2b}{\\beta} + N - 1`, to give an idea of the corresponding second dimensions :math:`M_{1,2}`.

        - If ``M1`` and ``M2`` are provided:

            In the :py:attr:`sampling_params` attribute, ``a`` and ``b`` are set to:
            ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and
            ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

        .. seealso::

            - :ref:`Tridiagonal matrix model <Jacobi_banded_matrix_model>` associated to the Jacobi ensemble
            - :cite:`KiNe04` Theorem 2
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "banded"

        if not (M1 and M2):  # default setting

            if self.beta > 0:
                M1 = 2 / self.beta * a + N - 1
                M2 = 2 / self.beta * b + N - 1

            else:
                M1, M2 = np.inf, np.inf

        elif (M1 >= N) and (M2 >= N):
            # all([var >= N for var in [M1, M2]]
            a = 0.5 * self.beta * (M1 - N + 1)
            b = 0.5 * self.beta * (M2 - N + 1)

        else:
            err_print = (
                "Must have M1, M2 >= N.",
                "Given: M1={}, M2={} and N={}".format(M1, M2, N),
            )
            raise ValueError(" ".join(err_print))

        params = {
            "a": a,
            "b": b,
            "N": N,
            "M1": M1,
            "M2": M2,
        }
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample i.i.d. Beta(a,b)
            # If M1,2 is used a, b = beta/2 (M1,2 - N + 1) = 0 => ERROR
            sampl = rng.beta(
                a=self.params["a"], b=self.params["b"], size=self.params["N"]
            )
        else:
            sampl = rm.mu_ref_beta_sampler_tridiag(
                a=self.params["a"],
                b=self.params["b"],
                beta=self.beta,
                size=self.params["N"],
                random_state=rng,
            )

        self.list_of_samples.append(sampl)
        return sampl

    def normalize_points(self, points):
        """No need to renormalize the points"""
        return points

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError("Empty `list_of_samples`, sample first!")
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample

        N, M_1, M_2 = [self.params[key] for key in ["N", "M1", "M2"]]

        fig, ax = plt.subplots(1, 1)
        # Title, answers Issue #33 raised by @adrienhardy
        str_ratio = ", ".join(
            [
                "with ratios",
                r"$M_1/N \approx {:.3f}$".format(M_1 / N),
                r"$M_2/N \approx {:.3f}$".format(M_2 / N),
            ]
        )
        str_beta = "" if self.beta > 0 else "with i.i.d. draws"
        title = "\n".join([self._str_title, " ".join([str_ratio, str_beta])])
        plt.title(title)

        if self.beta == 0:
            if normalization:
                # Display Beta(a,b) reference measure
                a, b = [self.params[key] for key in ["a", "b"]]
                x = np.linspace(0, 1, 100)
                ax.plot(
                    x,
                    stats.beta.pdf(x, a=a, b=b),
                    "r-",
                    lw=2,
                    alpha=0.6,
                    label=r"$\operatorname{{Beta}}({},{})$".format(a, b),
                )
        else:  # self.beta > 0
            if normalization:
                # Display the limiting distribution: Wachter law
                eps = 5e-3
                x = np.linspace(eps, 1.0 - eps, 500)
                ax.plot(
                    x,
                    rm.wachter_law(x, M_1, M_2, N),
                    "r-",
                    lw=2,
                    alpha=0.6,
                    label=r"$f_{Wachter}$",
                )

        if display_type == "scatter":
            ax.scatter(points, np.zeros_like(points), c="blue", label="sample")

        elif display_type == "hist":
            ax.hist(
                points, bins=30, density=1, facecolor="blue", alpha=0.5, label="hist"
            )
        else:
            pass

        plt.legend(loc="best", frameon=False)

    def plot(self, normalization=True):
        """Display the last realization of the :class:`JacobiBetaEnsemble` object

        :param normalization:
            When ``True``

            - If :py:attr:`beta` :math:`=0`, display the p.d.f. of the :math:`\\operatorname{Beta}(a, b)`
            - If :py:attr:`beta` :math:`>0`, display the limiting distribution, i.e., the Wachter distribution

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`hist`
            - :ref:`Full matrix model <Jacobi_full_matrix_model>` associated to the Jacobi ensemble
            - :ref:`Tridiagonal matrix model <Jacobi_banded_matrix_model>` associated to the Jacobi ensemble
        """

        self.__display_and_normalization("scatter", normalization)

    def hist(self, normalization=True):
        """Display the histogram of the last realization of the :class:`JacobiBetaEnsemble` object.

        :param normalization:
            When ``True``

            - If :py:attr:`beta` :math:`=0`, display the p.d.f. of the :math:`\\operatorname{Beta}(a, b)`
            - If :py:attr:`beta` :math:`>0`, display the limiting distribution, i.e., the Wachter distribution

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`plot`
            - :ref:`Full matrix model <Jacobi_full_matrix_model>` associated to the Jacobi ensemble
            - :ref:`Tridiagonal matrix model <Jacobi_banded_matrix_model>` associated to the Jacobi ensemble
        """
        self.__display_and_normalization("hist", normalization)
