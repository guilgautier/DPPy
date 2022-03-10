"""
- :class:`PoissonizedPlancherel` measure
- :func:`RSK`: Robinson-Schensted-Knuth correspondande
- :func:`xy_young_ru` young diagram -> russian convention coordinates
- :func:`limit_shape`
"""

from bisect import bisect_right

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import collections as mc

from dppy.utils import check_random_state


class PoissonizedPlancherel:
    """DPP on partitions associated to the Poissonized Plancherel measure

    :param theta:
        Poisson parameter i.e. expected length of the permutation

    :type theta:
        int, default 10

    .. seealso::

        - :cite:`Bor09` Section 6
        - :ref:`poissonized_plancherel_measure`
    """

    def __init__(self, theta=10):
        self.theta = theta

    def __str__(self):
        return "Poissonized Plancherel with parameter {}".format(self.theta)

    def sample(self, random_state=None):
        """Sample from the Poissonized Plancherel measure.

        :param random_state:
        :type random_state:
            None, np.random, int, np.random.RandomState
        """
        rng = check_random_state(random_state)

        size = rng.poisson(self.theta)
        sigma = rng.permutation(size)
        P, _ = RSK(sigma)

        sample = np.array([len(row) - i - 0.5 for i, row in enumerate(P)])
        return sample

    def plot(self, sample, ax=None):
        """Display the realization ``sample`` of the corresponding process."""

        if ax is None:
            fig, ax = plt.subplots(figsize=(19, 2))

        ax.scatter(sample, np.zeros_like(sample), color="blue", s=20)

        title = r"Realization of the Poissonized Plancherel process with parameter $\theta=${}".format(
            self.theta
        )
        plt.title(title)

        # Spine options
        ax.spines["bottom"].set_position("center")
        ax.spines["left"].set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Ticks options
        x_max = np.max(np.abs(sample)) + 0.5
        minor_ticks = np.arange(-x_max, x_max + 1)
        major_ticks = np.arange(-100, 100 + 1, 10)
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
        ax.set_xlim([-x_max - 2, x_max + 2])
        # ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

        return ax

    def plot_diagram(self, sample, normalize=False, ax=None):
        """Display the Young diagram (russian convention), the associated sample and potentially rescale the two to visualize the limit-shape theorem :cite:`Ker96`.
        The sample corresponds to the projection onto the real line of the descending surface edges.

        :param normalize:
            If ``normalize=True``, the Young diagram and the corresponding sample are scaled by a factor :math:`\sqrt{\theta}` and the limiting

        :type normalize:
            bool, default False

        .. seealso::

            - :py:meth:`sample`
            - :py:meth:`plot`
            - :cite:`Ker96`
        """

        sample_ = sample.copy()
        y_diag = (sample_ + np.arange(0.5, sample_.size)).astype(int)

        x_max = 1.1 * max(y_diag.size, y_diag[0])
        xy_young = xy_young_ru(y_diag)

        if normalize:
            norm = np.sqrt(self.theta)
            sample_ /= norm
            x_max /= norm
            xy_young /= norm

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 8))

        title = "\n".join(
            [
                r"Realization of the Poissonized Plancherel measure with parameter $\theta=${}".format(
                    self.theta
                ),
                "and the corresponding Young diagram (Russian conventions)",
            ]
        )
        plt.title(title)

        # Display corresponding sample
        ax.scatter(sample_, np.zeros_like(sample_), s=3, label="sample")
        # Display absolute value wedge
        ax.plot([-x_max, 0.0, x_max], [x_max, 0.0, x_max], c="k", lw=1)
        # Display young diagram in russian notation
        lc = mc.LineCollection(xy_young.reshape((-1, 2, 2)), color="k", linewidths=2)
        ax.add_collection(lc)
        # Display limit shape
        if normalize:
            x_lim_sh = np.linspace(-x_max, x_max, 100)
            ax.plot(x_lim_sh, limit_shape(x_lim_sh), c="r", label="limit shape")

        # Display stems linking sample on real line and descent in young diag
        # xy_y_diag = np.column_stack([y_diag,
        # np.arange(0.5, y_diag.size)]).dot(rot_45_and_scale.T)
        # if normalize:
        #     xy_y_diag /= np.sqrt(theta)
        # plt.scatter(xy_y_diag[:,0], np.zeros_like(y_diag), color='r')
        # plt.stem(xy_y_diag[:,0], xy_y_diag[:,1], linefmt='C0--', basefmt=' ')

        plt.legend(loc="best")
        plt.axis("equal")

        return ax


def RSK(sequence):
    """Apply Robinson-Schensted-Knuth correspondence on a sequence of reals, e.g. a permutation, and return the corresponding insertion and recording tableaux.

    :param sequence:
        Sequence of real numbers
    :type sequence:
        array_like

    :return:
        :math:`P, Q` insertion and recording tableaux
    :rtype:
        list

    .. seealso::

        `RSK Wikipedia <https://en.wikipedia.org/wiki/Robinson%E2%80%93Schensted%E2%80%93Knuth_correspondence>`_
    """

    P, Q = [], []  # Insertion/Recording tableau

    for it, x in enumerate(sequence, start=1):

        # Iterate along the rows of the tableau P to find a place for the bouncing x and record the position where it is inserted
        for row_P, row_Q in zip(P, Q):

            # If x finds a place at the end of a row of P
            if x >= row_P[-1]:
                row_P.append(x)  # add the element at the end of the row of P
                row_Q.append(it)  # record its position in the row of Q
                break
            else:
                # find place for x in the row of P to keep the row ordered
                ind_insert = bisect_right(row_P, x)
                # Swap x with the value in place
                x, row_P[ind_insert] = row_P[ind_insert], x

        # If no room for x at the end of any row of P create a new row
        else:
            P.append([x])
            Q.append([it])

    return P, Q


def xy_young_ru(young_diag):
    """Compute the xy coordinates of the boxes defining the young diagram, using the russian convention.

    :param young_diag:
        points
    :type  young_diag:
        array_like

    :return:
        :math:`\omega(x)`
    :rtype:
        array_like
    """

    def intertwine(arr_1, arr_2):
        inter = np.empty((arr_1.size + arr_2.size,), dtype=arr_1.dtype)
        inter[0::2], inter[1::2] = arr_1, arr_2
        return inter

    # horizontal lines
    x_hor = intertwine(np.zeros_like(young_diag), young_diag)
    y_hor = np.repeat(np.arange(1, young_diag.size + 1), repeats=2)

    # vertical lines
    uniq, ind = np.unique(young_diag[::-1], return_index=True)
    gaps = np.ediff1d(uniq, to_begin=young_diag[-1])

    x_vert = np.repeat(np.arange(1, 1 + gaps.sum()), repeats=2)
    y_vert = np.repeat(young_diag.size - ind, repeats=gaps)
    y_vert = intertwine(np.zeros_like(y_vert), y_vert)

    xy_young_fr = np.column_stack(
        [np.hstack([x_hor, x_vert]), np.hstack([y_hor, y_vert])]
    )

    rot_45_and_scale = np.array([[1.0, -1.0], [1.0, 1.0]])

    return xy_young_fr.dot(rot_45_and_scale.T)


def limit_shape(x):
    r"""Evaluate the limit-shape function :cite:`Ker96`

    .. math::

        \omega(x) =
        \begin{cases}
            |x|, &\text{if } |x|\geq 2\
            \frac{2}{\pi} \left(x \arcsin\left(\frac{x}{2}\right) + \sqrt{4-x^2} \right) &\text{otherwise } \end{cases}

    :param x:
        points
    :type x:
        array_like

    :return:
        :math:`\omega(x)`
    :rtype:
        array_like

    .. seealso::

        - :func:`plot_diagram <plot_diagram>`
        - :cite:`Ker96`
    """
    w_x = np.abs(x)
    mask = w_x < 2.0
    w_x[mask] = x[mask] * np.arcsin(0.5 * x[mask]) + np.sqrt(4.0 - x[mask] ** 2)
    w_x[mask] *= 2.0 / np.pi
    return w_x
