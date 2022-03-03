import matplotlib.pyplot as plt
import numpy as np

import dppy.random_matrices as rm
from dppy.beta_ensembles.beta_ensembles import AbstractBetaEnsemble
from dppy.utils import check_random_state


class GinibreEnsemble(AbstractBetaEnsemble):
    """Ginibre Ensemble object

    .. seealso::

        - :ref:`Full matrix model <ginibre_full_matrix_model>` associated to the Ginibre ensemble
    """

    def __init__(self, beta=2):

        if beta != 2:
            err_print = (
                "Ginibre ensemble is only available for `beta`=2.",
                "Given {}".format(beta),
            )
            raise ValueError(" ".join(err_print))
        super().__init__(beta=beta)

        params = {"size_N": 10}
        self.params.update(params)

    def sample_full_model(self, size_N=10, random_state=None):
        """Sample from :ref:`full matrix model <Ginibre_full_matrix_model>` associated to the Ginibre ensemble. Only available for :py:attr:`beta` :math:`=2`

        :param size_N:
            Number :math:`N` of points, i.e., size of the matrix to be diagonalized
        :type size_N:
            int, default :math:`10`

        .. seealso::

            - :ref:`Full matrix model <ginibre_full_matrix_model>` associated to the Ginibre ensemble
        """
        rng = check_random_state(random_state)

        self.params.update({"size_N": size_N})
        sampl = rm.ginibre_sampler_full(N=self.params["size_N"], random_state=rng)

        self.list_of_samples.append(sampl)
        return sampl

    def sample_banded_model(self, *args, **kwargs):
        """No banded model is known for Ginibre, use :py:meth:`sample_full_model`"""
        raise NotImplementedError(
            "No banded model is known for Ginibre, use `sample_full_model`"
        )

    def normalize_points(self, points):
        """Normalize points to concentrate in the unit disk.

            .. math::

                x \\mapsto \\frac{x}{\\sqrt{N}}

        .. seealso::

            - :py:meth:`plot`
            - :py:meth:`hist`
        """

        return points / np.sqrt(self.params["size_N"])

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError("Empty `list_of_samples`, sample first!")
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample
            if normalization:
                points = self.normalize_points(points)

        fig, ax = plt.subplots(1, 1)

        title = self._str_title

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
            title = "\n".join([title, "Histogram of the modulus of each points"])

            if normalization:
                ax.plot([0, 1, 1], [0, 2, 0], color="r")

            bins = np.linspace(0, 1, 20)
            ax.hist(
                np.abs(points),
                bins=bins,
                density=1,
                facecolor="blue",
                alpha=0.5,
                label="hist",
            )
        else:
            pass

        plt.title(title)

        # plt.legend(loc='best', frameon=False)

    def plot(self, normalization=True):
        """Display the last realization of the :class:`GinibreEnsemble` object

        :param normalization:
            When ``True``, the points are first normalized so as to concentrate in the unit disk (see :py:meth:`normalize_points`) and the unit circle is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`normalize_points`
            - :py:meth:`sample_full_model`
            - :ref:`Full matrix model <ginibre_full_matrix_model>` associated to the Ginibre ensemble ensemble
        """

        self.__display_and_normalization("scatter", normalization)

    def hist(self, normalization=True):
        """Display the histogram of the radius of the points the last realization of the :class:`GinibreEnsemble` object

        :param normalization:
            When ``True``, the points are first normalized so as to concentrate in the unit disk (see :py:meth:`normalize_points`) and the limiting density :math:`2r 1_{[0, 1]}(r)` of the radii is displayed

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`normalize_points`
            - :py:meth:`sample_full_model`
            - :ref:`Full matrix model <ginibre_full_matrix_model>` associated to the Ginibre ensemble ensemble
        """

        self.__display_and_normalization("hist", normalization)
