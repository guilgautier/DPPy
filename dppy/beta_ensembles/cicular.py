import matplotlib.pyplot as plt
import numpy as np

import dppy.random_matrices as rm
from dppy.beta_ensembles.beta_ensembles import AbstractBetaEnsemble
from dppy.utils import check_random_state


class CircularBetaEnsemble(AbstractBetaEnsemble):
    """Circular Ensemble object

    .. seealso::

        - :ref:`Full matrix model <circular_full_matrix_model>` associated to the Circular ensemble
        - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` associated to the Circular ensemble
    """

    def __init__(self, beta=2):

        if not isinstance(beta, int):
            raise ValueError("`beta` must be int >0. Given: {}".format(beta))
        super().__init__(beta=beta)

        params = {"size_N": 10}
        self.params.update(params)

    def sample_full_model(self, size_N=10, haar_mode="Hermite", random_state=None):
        """Sample from :ref:`tridiagonal matrix model <Circular_full_matrix_model>` associated to the Circular ensemble. Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. uniform points on the unit circle

        :param size_N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized
        :type size_N:
            int, default :math:`10`

        :param haar_mode:
            Sample Haar measure i.e. uniformly on the orthogonal/unitary/symplectic group using:
            - 'QR',
            - 'Hermite'
        :type haar_mode:
            str, default 'hermite'

        .. seealso::

            - :ref:`Full matrix model <circular_full_matrix_model>` associated to the Circular ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "full"
        params = {"size_N": size_N, "haar_mode": haar_mode}
        self.params.update(params)

        if self.beta == 0:  # i.i.d. points uniformly on the circle
            # Answer issue #28 raised by @rbardenet
            sampl = np.exp(2 * 1j * np.pi * rng.rand(params["size_N"]))
        else:
            sampl = rm.circular_sampler_full(
                N=self.params["size_N"],
                beta=self.beta,
                haar_mode=self.params["haar_mode"],
                random_state=rng,
            )

        self.list_of_samples.append(sampl)
        return sampl

    def sample_banded_model(self, size_N=10, random_state=None):
        """Sample from :ref:`Quindiagonal matrix model <Circular_banded_matrix_model>` associated to the Circular Ensemble.
        Available for :py:attr:`beta` :math:`\\in\\mathbb{N}^*`, and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. uniform points on the unit circle

        :param size_N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized
        :type size_N:
            int, default :math:`10`

        .. note::

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` parameter.

        .. seealso::

            - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` associated to the Circular ensemble
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = "banded"
        params = {"size_N": size_N}
        self.params.update(params)

        if self.beta == 0:  # i.i.d. points uniformly on the circle
            # Answer issue #28 raised by @rbardenet
            sampl = np.exp(2 * 1j * np.pi * rng.rand(params["size_N"]))
        else:
            sampl = rm.mu_ref_unif_unit_circle_sampler_quindiag(
                beta=self.beta, size=self.params["size_N"], random_state=rng
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

        fig, ax = plt.subplots(1, 1)

        # Title
        samp_mod = ""

        if self.beta == 0:
            samp_mod = r"i.i.d samples from $\mathcal{U}_{[0,2\pi]}$"
        elif self.sampling_mode == "full":
            samp_mod = "full matrix model with haar_mode={}".format(
                self.params["haar_mode"]
            )
        else:  # self.sampling_mode == 'banded':
            samp_mod = "quindiag model"

        title = "\n".join([self._str_title, "using {}".format(samp_mod)])
        plt.title(title)

        if display_type == "scatter":

            if normalization:
                # Draw unit circle
                unit_circle = plt.Circle((0, 0), 1, color="r", fill=False)
                ax.add_artist(unit_circle)

            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.set_aspect("equal")
            ax.scatter(points.real, points.imag, c="blue", label="sample")

        elif display_type == "hist":
            points = np.angle(points)

            if normalization:
                # Uniform distribution on [0, 2pi]
                ax.axhline(y=1 / (2 * np.pi), color="r", label=r"$\frac{1}{2\pi}$")

            ax.hist(
                points, bins=30, density=1, facecolor="blue", alpha=0.5, label="hist"
            )
        else:
            pass

        plt.legend(loc="best", frameon=False)

    def plot(self, normalization=True):
        """Display the last realization of the :class:`CircularBetaEnsemble` object.

        :param normalization:
            When ``True``, the unit circle is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`hist`
            - :ref:`Full matrix model <circular_full_matrix_model>` associated to the Circular ensemble
            - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` associated to the Circular ensemble
        """

        self.__display_and_normalization("scatter", normalization)

    def hist(self, normalization=True):
        """Display the histogram of the angles :math:`\\theta_{1}, \\dots, \\theta_{N}` associated to the last realization :math:`\\left\\{ e^{i \\theta_{1}}, \\dots, e^{i \\theta_{N}} \\right\\}`of the :class:`CircularBetaEnsemble` object.

        :param normalization:
            When ``True``, the limiting distribution of the angles, i.e., the uniform distribution in :math:`[0, 2\\pi]` is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`plot`
            - :ref:`Full matrix model <circular_full_matrix_model>` associated to the Circular ensemble
            - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` associated to the Circular ensemble
        """
        self.__display_and_normalization("hist", normalization)
