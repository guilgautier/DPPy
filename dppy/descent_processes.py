# coding: utf8
"""Descent processes:

- :class:`CarriesProcess`
- :class:`DescentProcess`
- :class:`VirtualDescentProcess`
"""

import matplotlib.pyplot as plt
import numpy as np

from dppy.utils import check_random_state


class AbstractDescent:
    @property
    def marginal_descent_probability(self):
        r"""Marginal probability :math:`\mathbb{P}[ X_{i} < X_{i-1} ]` that a decent occurs at any index :math:`i \geq 1`."""
        return 0.5

    def sample(self, size=100, random_state=None):
        """Sample from corresponding descent process."""
        rng = check_random_state(random_state)
        p = self.marginal_descent_probability
        return rng.rand(size) < p

    def plot(self, sample, ax=None):
        """Display the realization ``sample`` of the corresponding descent process.

        .. seealso::

            - :py:meth:`sample`
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(19, 2))

        size = len(sample)
        ax.scatter(
            x=np.arange(size),
            y=np.zeros(size),
            color=np.where(sample, "black", "white"),
            edgecolors="gray",
            linewidths=0.5,
            s=20,
        )

        title = "Realization of the {}".format(str(self))
        plt.title(title)

        # Spine options
        ax.spines["bottom"].set_position("center")
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Ticks options
        minor_ticks = np.arange(0, size + 1)
        major_ticks = np.arange(0, size + 1, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticklabels(major_ticks, fontsize=15)
        ax.xaxis.set_ticks_position("bottom")

        ax.tick_params(
            axis="y",  # changes apply to the y-axis
            which="both",  # both major and minor ticks are affected
            left=False,  # ticks along the left edge are off
            right=False,  # ticks along the right edge are off
            labelleft=False,
        )  # labels along the left edge are off

        ax.xaxis.grid(True)
        ax.set_xlim([-1, size + 1])
        # ax.legend(bbox_to_anchor=(0, 0.85), frameon=False, prop={"size": 15})
        return ax


class CarriesProcess(AbstractDescent):
    """DPP on :math:`\{1, \dots, N-1\}` (with a non symmetric kernel) derived from the cumulative sum of :math:`N` i.i.d. digits in :math:`\{0, \dots, b-1\}`.

    :param base:
        Base/radix

    :type base:
        int, default 10

    .. seealso::

        - :cite:`BoDiFu10`
        - :ref:`carries_process`
    """

    def __init__(self, base=10):
        super().__init__()
        self.base = base

    def __str__(self):
        b = self.base
        p = self.marginal_descent_probability
        return r"Carries process in base {}, $P[D_i = 1] = {}$".format(b, p)

    @property
    def marginal_descent_probability(self):
        return 0.5 * (1 - 1 / self.base)

    def sample(self, size=100, random_state=None):
        r"""Record where carries occur in the cumulative sum (in base :math:`b`) of a sequence of i.i.d. digits from :math:`\{0, \dots, b-1\}`.

        :param size:
            size of the sample.
        :type size:
            int

        :return: vector indicating if descent occurred (True).
        :rtype: np.ndarray
        """
        rng = check_random_state(random_state)
        remainder = rng.randint(0, self.base, size)
        np.cumsum(remainder, out=remainder)
        np.mod(remainder, self.base, out=remainder)
        carries = np.zeros_like(remainder, dtype=bool)
        carries[1:] = remainder[1:] < remainder[:-1]
        return carries


class DescentProcess(AbstractDescent):
    """DPP on :math:`\{0, \dots, N-1\}` associated to the descent process on the symmetric group :math:`\mathfrak{S}_N`.

    .. seealso::

        - :cite:`BoDiFu10`
        - :ref:`descent_process`
    """

    def __init__(self):
        super().__init__()

    def __str__(self):
        p = self.marginal_descent_probability
        return r"Descent process, $P[D_i = 1] = {}$".format(p)

    @property
    def marginal_descent_probability(self):
        return 0.5

    def sample(self, size=100, random_state=None):
        """Record where descents occur in uniform random permutation :math:`\sigma \in \mathfrak{S}_N`, i.e., :math:`\{ i ~;~ \sigma_i > \sigma_{i+1} \}`.

        :param size:
            size of the sample, equal to :math:`N`.
        :type size:
            int

        :return: vector indicating if descent occurred (True).
        :rtype: np.ndarray
        """
        rng = check_random_state(random_state)
        sigma = rng.permutation(size)
        descent = np.zeros_like(sigma, dtype=bool)
        descent[1:] = sigma[1:] < sigma[:-1]
        return descent


class VirtualDescentProcess(AbstractDescent):
    """This is a DPP on :math:`\{0, \dots, N-1\}` with a non symmetric kernel appearing as a limit of the descent process on the symmetric group :math:`\mathfrak{S}_N`.

    .. seealso::

        - :cite:`Kam18`
        - :ref:`limiting_descent_process`
        - :class:`DescentProcess`
    """

    def __init__(self, x0=0.5):
        super().__init__()
        assert 0 <= x0 <= 1
        self.x0 = x0

    def __str__(self):
        p = self.marginal_descent_probability
        return r"Virtual permutation descent process, $P[D_i = 1] = {}$".format(p)

    @property
    def marginal_descent_probability(self):
        return 0.5 * (1.0 - self.x0 ** 2)

    def sample(self, size=100, random_state=None):
        """Record the descents of a virtual permutation of size`=size`, see :cite:`Kam18`.

        TODO ask @kammmoun to complete the docsting and check that the code is correct.

        :param size:
            size of the virtual permutation.
        :type size:
            int

        :return: boolean vector where each entry indicates whether a descent occured at the corresponding index.
        :rtype: np.ndarray
        """

        rng = check_random_state(random_state)
        sigma = rng.permutation(size)
        X = sigma[1:] < sigma[:-1]  # Record the descents in permutation
        Y = rng.binomial(n=2, p=self.x0, size=size) != 1
        virtual_descent = np.zeros_like(Y, dtype=bool)
        virtual_descent[1:] = np.array(
            [
                (not Y[i] and Y[i + 1]) or (not Y[i] and not Y[i + 1] and x)
                for i, x in enumerate(X)
            ]
        )
        return virtual_descent
