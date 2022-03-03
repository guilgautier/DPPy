import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

import dppy.random_matrices as rm
from dppy.beta_ensembles.abstract_beta_ensemble import AbstractBetaEnsemble
from dppy.utils import check_random_state


class HermiteBetaEnsemble(AbstractBetaEnsemble):
    """Hermite Ensemble object

    .. seealso::

        - :ref:`Full matrix model <hermite_full_matrix_model>` associated to the Hermite ensemble
        - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` associated to the Hermite ensemble
    """

    def __init__(self, beta=2):

        super().__init__(beta=beta)

        # Default parameters for ``loc`` and ``scale`` correspond to the
        # reference measure N(0,2) in the full matrix model
        params = {"loc": 0.0, "scale": np.sqrt(2.0), "N": 10}
        self.params.update(params)

    def sample_full_model(self, N=10, random_state=None):
        """Sample from :ref:`full matrix model <Hermite_full_matrix_model>` associated to the Hermite ensemble.
        Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}`
        and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the Gaussian :math:`\\mathcal{N}(\\mu,\\sigma^2)` reference measure

        :param N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized
        :type N:
            int, default :math:`10`

        .. note::

            The reference measure associated with the :ref:`full matrix model <hermite_full_matrix_model>` is :math:`\\mathcal{N}(0,2)`.
            For this reason, in the :py:attr:`sampling_params` attribute, the values of the parameters are set to ``loc``:math:`=0` and ``scale``:math:`=\\sqrt{2}`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``N`` parameter.

        .. seealso::

            - :ref:`Full matrix model <hermite_full_matrix_model>` associated to the Hermite ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "full"
        params = {"loc": 0.0, "scale": np.sqrt(2.0), "N": N}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample N i.i.d. gaussian N(0,2)
            sampl = rng.normal(
                loc=self.params["loc"],
                scale=self.params["scale"],
                size=self.params["N"],
            )
        else:  # if beta > 0
            sampl = rm.hermite_sampler_full(
                N=self.params["N"], beta=self.beta, random_state=rng
            )

        self.list_of_samples.append(sampl)
        return sampl

    def sample_banded_model(self, loc=0.0, scale=np.sqrt(2.0), N=10, random_state=None):
        """Sample from :ref:`tridiagonal matrix model <hermite_banded_matrix_model>` associated to the Hermite Ensemble.
        Available for :py:attr:`beta` :math:`>0` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the Gaussian :math:`\\mathcal{N}(\\mu,\\sigma^2)` reference measure

        :param loc:
            Mean :math:`\\mu` of the Gaussian :math:`\\mathcal{N}(\\mu, \\sigma^2)`
        :type loc:
            float, default :math:`0`

        :param scale:
            Standard deviation :math:`\\sigma` of the Gaussian :math:`\\mathcal{N}(\\mu, \\sigma^2)`
        :type scale:
            float, default :math:`\\sqrt{2}`

        :param N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized
        :type N:
            int, default :math:`10`

        .. note::

            The reference measure associated with the :ref:`full matrix model <hermite_full_matrix_model>` is :math:`\\mathcal{N}(0,2)`.
            For this reason, in the :py:attr:`sampling_params` attribute, the default values are set to ``loc``:math:`=0` and ``scale``:math:`=\\sqrt{2}`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``N`` parameter.

        .. seealso::

            - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` associated to the Hermite ensemble
            - :cite:`DuEd02` II-C
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "banded"
        params = {"loc": loc, "scale": scale, "N": N}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample N i.i.d. gaussian N(mu, sigma^2)
            sampl = rng.normal(
                loc=self.params["loc"],
                scale=self.params["scale"],
                size=self.params["N"],
            )
        else:  # if beta > 0
            sampl = rm.mu_ref_normal_sampler_tridiag(
                loc=self.params["loc"],
                scale=self.params["scale"],
                beta=self.beta,
                size=self.params["N"],
                random_state=rng,
            )

        self.list_of_samples.append(sampl)
        return sampl

    def normalize_points(self, points):
        """Normalize points obtained after sampling to fit the limiting distribution, i.e., the semi-circle

        .. math::

            f(x) = \\frac{1}{2\\pi} \\sqrt{4-x^2}

        :param points:
            A sample from Hermite ensemble, accessible through the :py:attr:`list_of_samples` attribute
        :type points:
            array_like

        - If sampled using :py:meth:`sample_banded_model` with reference measure :math:`\\mathcal{N}(\\mu,\\sigma^2)`

            1. Normalize the points to fit the p.d.f. of :math:`\\mathcal{N}(0,2)` reference measure of the :ref:`full matrix model <hermite_full_matrix_model>`

                .. math::

                    x \\mapsto \\sqrt{2}\\frac{x-\\mu}{\\sigma}

            2. If :py:attr:`beta` :math:`>0`, normalize the points to fit the semi-circle distribution

                .. math::

                    x \\mapsto \\frac{x}{\\beta N}

                Otherwise if :py:attr:`beta` :math:`=0` do nothing more

        - If sampled using :py:meth:`sample_full_model`, apply 2. above

        .. note::

            This method is called in :py:meth:`plot` and :py:meth:`hist` when ``normalization=True``
        """

        if self.sampling_mode == "banded":
            # Normalize to fit N(0,2) ref measure of the full matrix model
            points -= self.params["loc"]
            points /= np.sqrt(0.5) * self.params["scale"]
        else:  # 'full'
            pass

        if self.beta > 0:
            # Normalize to fit the semi-circle distribution
            points /= np.sqrt(self.beta * self.params["N"])
        else:
            pass

        return points

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError("Empty `list_of_samples`,sample first!")
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample
            if normalization:
                points = self.normalize_points(points)

        fig, ax = plt.subplots(1, 1)
        # Title
        str_beta = "" if self.beta > 0 else "with i.i.d. draws"
        title = "\n".join([self._str_title, str_beta])
        plt.title(title)

        if self.beta == 0:
            if normalization:
                # Display N(0,2) reference measure of the full matrix model
                mu, sigma = 0.0, np.sqrt(2.0)
                x = mu + 3.5 * sigma * np.linspace(-1, 1, 100)
                ax.plot(
                    x,
                    stats.norm.pdf(x, mu, sigma),
                    "r-",
                    lw=2,
                    alpha=0.6,
                    label=r"$\mathcal{{N}}(0,2)$",
                )
            else:
                pass

        else:  # self.beta > 0
            if normalization:
                # Display the limiting distribution: semi-circle law
                x = np.linspace(-2, 2, 100)
                ax.plot(
                    x,
                    rm.semi_circle_law(x),
                    "r-",
                    lw=2,
                    alpha=0.6,
                    label=r"$f_{semi-circle}$",
                )
            else:
                pass

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
        """Display the last realization of the :class:`HermiteBetaEnsemble` object

        :param normalization:
            When ``True``, the points are first normalized (see :py:meth:`normalize_points`) so that they concentrate as

            - If :py:attr:`beta` :math:`=0`, the :math:`\\mathcal{N}(0, 2)` reference measure associated to full :ref:`full matrix model <hermite_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0`, the limiting distribution, i.e., the semi-circle distribution

            in both cases, the corresponding p.d.f. is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`hist`
            - :ref:`Full matrix model <hermite_full_matrix_model>` associated to the Hermite ensemble
            - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` associated to the Hermite ensemble
        """

        self.__display_and_normalization("scatter", normalization)

    def hist(self, normalization=True):
        """Display the histogram of the last realization of the :class:`HermiteBetaEnsemble` object.

        :param normalization:
            When ``True``, the points are first normalized (see :py:meth:`normalize_points`) so that they concentrate as

            - If :py:attr:`beta` :math:`=0`, the :math:`\\mathcal{N}(0, 2)` reference measure associated to full :ref:`full matrix model <hermite_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0`, the limiting distribution, i.e., the semi-circle distribution

            in both cases, the corresponding p.d.f. is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`plot`
            - :ref:`Full matrix model <hermite_full_matrix_model>` associated to the Hermite ensemble
            - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` associated to the Hermite ensemble
        """
        self.__display_and_normalization("hist", normalization)
