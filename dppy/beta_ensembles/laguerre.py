import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import dppy.random_matrices as rm
from dppy.beta_ensembles.beta_ensembles import AbstractBetaEnsemble
from dppy.utils import check_random_state


class LaguerreBetaEnsemble(AbstractBetaEnsemble):
    """Laguerre Ensemble object

    .. seealso::

        - :ref:`Full matrix model <laguerre_full_matrix_model>` associated to the Laguerre ensemble
        - :ref:`Tridiagonal matrix model <laguerre_banded_matrix_model>` associated to the Laguerre ensemble
    """

    def __init__(self, beta=2):

        super().__init__(beta=beta)

        params = {"shape": 0.0, "scale": 2.0, "size_N": 10, "size_M": None}
        self.params.update(params)

    def sample_full_model(self, size_N=10, size_M=100, random_state=None):
        """Sample from :ref:`full matrix model <Laguerre_full_matrix_model>` associated to the Laguerre ensemble. Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\Gamma(k,\\theta)` reference measure

        :param size_N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized.
            First dimension of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_full_matrix_model>`.
        :type size_N:
            int, default :math:`10`

        :param size_M:
            Second dimension :math:`M` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_full_matrix_model>`.

        :type size_M:
            int, default :math:`100`

        .. note::

            The reference measure associated with the :ref:`full matrix model <laguerre_full_matrix_model>` is :math:`\\Gamma\\left(\\frac{\\beta}{2}(M-N+1), 2\\right)`.
            For this reason, in the :py:attr:`sampling_params`, the values of the parameters are set to ``shape``:math:`=\\frac{\\beta}{2}(M-N+1)` and ``scale``:math:`=2`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` and ``size_M`` parameters.

        .. seealso::

            - :ref:`Full matrix model <Laguerre_full_matrix_model>` associated to the Laguerre ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "full"

        if size_M >= size_N:
            # Define the parameters of the associated Gamma distribution
            shape, scale = 0.5 * self.beta * (size_M - size_N + 1), 2.0
        else:
            err_print = (
                "Must have M >= N.",
                "Given: M={} < N={}".format(size_M, size_N),
            )
            raise ValueError(" ".join(err_print))

        params = {"shape": shape, "scale": scale, "size_N": size_N, "size_M": size_M}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # rng.gamma(shape=0,...) when doesn't return error! contrary to sp.stats.gamma(a=0).rvs(), see https://github.com/numpy/numpy/issues/12367
            # sampl = stats.gamma.rvs(a=params['shape'], loc=0.0, scale=params['scale'],                                       size=params['size_N'])
            if params["shape"] > 0:
                sampl = rng.gamma(
                    shape=self.params["shape"],
                    scale=self.params["scale"],
                    size=self.params["size_N"],
                )
            else:
                err_print = ("shape<=0.", "Here beta=0, hence shape=beta/2*(M-N+1)=0")
                raise ValueError(" ".join(err_print))

        else:  # if beta > 0
            sampl = rm.laguerre_sampler_full(
                M=self.params["size_M"],
                N=self.params["size_N"],
                beta=self.beta,
                random_state=rng,
            )

        self.list_of_samples.append(sampl)
        return sampl

    def sample_banded_model(
        self, shape=1.0, scale=2.0, size_N=10, size_M=None, random_state=None
    ):
        """Sample from :ref:`tridiagonal matrix model <Laguerre_banded_matrix_model>` associated to the Laguerre ensemble. Available for :py:attr:`beta` :math:`>0` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\Gamma(k,\\theta)` reference measure

        :param shape:
            Shape parameter :math:`k` of :math:`\\Gamma(k, \\theta)` reference measure
        :type shape:
            float, default :math:`1`

        :param scale:
            Scale parameter :math:`\\theta` of :math:`\\Gamma(k, \\theta)` reference measure
        :type scale:
            float, default :math:`2.0`

        :param size_N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized.
            Equivalent to the first dimension :math:`N` of the matrix used to form the covariance matrix in the :ref:`full matrix model <laguerre_full_matrix_model>`.
        :type size_N:
            int, default :math:`10`

        :param size_M:
            Equivalent to the second dimension :math:`M` of the matrix used to form the covariance matrix in the :ref:`full matrix model <laguerre_full_matrix_model>`.

        :type size_M:
            int, default None

        - If ``size_M`` is not provided:

            In the :py:attr:`sampling_params`, ``size_M`` is set to
            ``size_M``:math:`= \\frac{2k}{\\beta} + N - 1`, to give an idea of the corresponding second dimension :math:`M`.


        - If ``size_M`` is provided:

            In the :py:attr:`sampling_params`, ``shape`` and ``scale`` are set to:
            ``shape``:math:`=\\frac{1}{2} \\beta (M-N+1)` and ``scale``:math:`=2`


        .. note::

            The reference measure associated with the :ref:`full matrix model <laguerre_full_matrix_model>` is :math:`\\Gamma\\left(\\frac{\\beta}{2}(M-N+1), 2\\right)`. This explains the role of the ``size_M`` parameter.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` and ``size_M`` parameters

        .. seealso::

            - :ref:`Tridiagonal matrix model <Laguerre_banded_matrix_model>` associated to the Laguerre ensemble
            - :cite:`DuEd02` III-B
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "banded"

        if not size_M:  # Default setting
            if self.beta > 0:
                size_M = 2 / self.beta * shape + size_N - 1
            else:
                size_M = np.inf

        elif size_M >= size_N:
            # define the parameters of the associated Gamma distribution
            shape, scale = 0.5 * self.beta * (size_M - size_N + 1), 2.0

        else:
            err_print = (
                "Must have M >= N.",
                "Given: M={} < N={}".format(size_M, size_N),
            )
            raise ValueError(" ".join(err_print))

        params = {"shape": shape, "scale": scale, "size_N": size_N, "size_M": size_M}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # rng.gamma(shape=0,...) when doesn't return error! contrary to sp.stats.gamma(a=0).rvs(), see https://github.com/numpy/numpy/issues/12367
            # sampl = stats.gamma.rvs(a=params['shape'], loc=0.0, scale=params['scale'], size=params['size_N'])
            if params["shape"] > 0:
                sampl = rng.gamma(
                    shape=self.params["shape"],
                    scale=self.params["scale"],
                    size=self.params["size_N"],
                )
            else:
                err_print = ("shape<=0.", "Here beta=0, hence shape=beta/2*(M-N+1)=0")
                raise ValueError(" ".join(err_print))

        else:  # if beta > 0
            sampl = rm.mu_ref_gamma_sampler_tridiag(
                shape=self.params["shape"],
                scale=self.params["scale"],
                beta=self.beta,
                size=self.params["size_N"],
                random_state=rng,
            )

        self.list_of_samples.append(sampl)
        return sampl

    def normalize_points(self, points):
        """Normalize points obtained after sampling to fit the limiting distribution, i.e., the Marcenko-Pastur distribution

        .. math::

            \\frac{1}{2\\pi}
            \\frac{\\sqrt{(\\lambda_+-x)(x-\\lambda_-)}}{cx}
            1_{[\\lambda_-,\\lambda_+]}
            dx

        where :math:`c = \\frac{M}{N}` and :math:`\\lambda_\\pm = (1\\pm\\sqrt{c})^2`

        :param points:
            A sample from Laguerre ensemble, accessible through the :py:attr:`list_of_samples` attribute
        :type points:
            array_like

        - If sampled using :py:meth:`sample_banded_model` with reference measure :math:`\\Gamma(k,\\theta)`

            .. math::

                x \\mapsto \\frac{2x}{\\theta}
                \\quad \\text{and} \\quad
                x \\mapsto \\frac{x}{\\beta M}

        - If sampled using :py:meth:`sample_full_model`

            .. math::

                x \\mapsto \\frac{x}{\\beta M}

        .. note::

            This method is called in :py:meth:`plot` and :py:meth:`hist` when ``normalization=True``.
        """

        if self.sampling_mode == "banded":
            # Normalize to fit Gamma(*,2) refe measure of the full matrix model
            points /= 0.5 * self.params["scale"]
        else:  # self.sampling_mode == 'full':
            pass

        if self.beta > 0:
            # Normalize to fit the Marcenko-Pastur distribution
            points /= self.beta * self.params["size_M"]
        else:
            pass
            # set a warning, won't concentrate as semi-circle, they are i.i.d.

        return points

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError("Empty `list_of_samples`, sample first!")
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample
            if normalization:
                points = self.normalize_points(points)

        N, M = [self.params[key] for key in ["size_N", "size_M"]]

        fig, ax = plt.subplots(1, 1)
        # Title
        str_ratio = r"with ratio $M/N \approx {:.3f}$".format(M / N)
        # Answers Issue #33 raised by @adrienhardy
        str_beta = "" if self.beta > 0 else "with i.i.d. draws"
        title = "\n".join([self._str_title, " ".join([str_ratio, str_beta])])
        plt.title(title)

        if self.beta == 0:
            if normalization:
                # Display Gamma(k,2) reference measure of the full matrix model
                k, theta = self.params["shape"], 2
                x = np.linspace(0, np.max(points) + 4, 100)
                ax.plot(
                    x,
                    stats.gamma.pdf(x, a=k, loc=0.0, scale=theta),
                    "r-",
                    lw=2,
                    alpha=0.6,
                    label=r"$\Gamma({},{})$".format(k, theta),
                )

        else:  # self.beta > 0
            if normalization:
                # Display the limiting distribution: Marcenko-Pastur
                x = np.linspace(1e-2, np.max(points) + 0.3, 100)
                ax.plot(
                    x,
                    rm.marcenko_pastur_law(x, M, N),
                    "r-",
                    lw=2,
                    alpha=0.6,
                    label=r"$f_{Marcenko-Pastur}$",
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
        """Display the last realization of the :class:`LaguerreBetaEnsemble` object

        :param normalization:
            When ``True``, the points are first normalized (see :py:meth:`normalize_points`) so that they concentrate as

            - If :py:attr:`beta` :math:`=0`, the :math:`\\Gamma(k, 2)` reference measure associated to full :ref:`full matrix model <laguerre_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0`, the limiting distribution, i.e., the Marcenko-Pastur distribution

            in both cases the corresponding p.d.f. is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`hist`
            - :ref:`Full matrix model <Laguerre_full_matrix_model>` associated to the Laguerre ensemble
            - :ref:`Tridiagonal matrix model <Laguerre_banded_matrix_model>` associated to the Laguerre ensemble
        """

        self.__display_and_normalization("scatter", normalization)

    def hist(self, normalization=True):
        """Display the histogram of the last realization of the :class:`LaguerreBetaEnsemble` object.

        :param normalization:
            When ``True``, the points are first normalized (see :py:meth:`normalize_points`) so that they concentrate as

            - If :py:attr:`beta` :math:`=0`, the :math:`\\Gamma(k, 2)` reference measure associated to full :ref:`full matrix model <laguerre_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0`, the limiting distribution, i.e., the Marcenko-Pastur distribution

            in both cases the corresponding p.d.f. is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`plot`
            - :ref:`Full matrix model <Laguerre_full_matrix_model>` associated to the Laguerre ensemble
            - :ref:`Tridiagonal matrix model <Laguerre_banded_matrix_model>` associated to the Laguerre ensemble
        """
        self.__display_and_normalization("hist", normalization)
