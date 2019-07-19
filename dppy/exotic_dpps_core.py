# coding: utf8
""" Core functions for

- Uniform spanning trees

    * :func:`ust_sampler_wilson`
    * :func:`ust_sampler_aldous_broder`:

- Descent procresses :class:`Descent`:

    * :func:`uniform_permutation`

- :class:`PoissonizedPlancherel` measure

    * :func:`uniform_permutation`
    * :func:`RSK`: Robinson-Schensted-Knuth correspondande
    * :func:`xy_young_ru` young diagram -> russian convention coordinates
    * :func:`limit_shape`

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/exotic_dpps/index.html>`_
"""

import functools  # used for decorators to pass docstring

import numpy as np

# For Uniform Spanning Trees
import networkx as nx
from itertools import chain  # create graph edges from path

# For class PoissonizedPlancherel
from bisect import bisect_right  # for RSK

from dppy.utils import check_random_state


def ust_sampler_wilson(list_of_neighbors, root=None,
                       random_state=None):

    rng = check_random_state(random_state)

    # Initialize the tree
    wilson_tree_graph = nx.Graph()
    nb_nodes = len(list_of_neighbors)

    # Initialize the root, if root not specified start from any node
    n0 = root if root else rng.choice(nb_nodes)  # size=1)[0]
    # -1 = not visited / 0 = in path / 1 = in tree
    state = -np.ones(nb_nodes, dtype=int)
    state[n0] = 1
    nb_nodes_in_tree = 1

    path, branches = [], []  # branches of tree, temporary path

    while nb_nodes_in_tree < nb_nodes:  # |Tree| = |V| - 1

        # visit a neighbor of n0 uniformly at random
        n1 = rng.choice(list_of_neighbors[n0])  # size=1)[0]

        if state[n1] == -1:  # not visited => continue the walk

            path.append(n1)  # add it to the path
            state[n1] = 0  # mark it as in the path
            n0 = n1  # continue the walk

        if state[n1] == 0:  # loop on the path => erase the loop

            knot = path.index(n1)  # find 1st appearence of n1 in the path
            nodes_loop = path[knot + 1:]  # identify nodes forming the loop
            del path[knot + 1:]  # erase the loop
            state[nodes_loop] = -1  # mark loopy nodes as not visited
            n0 = n1  # continue the walk

        elif state[n1] == 1:  # hits the tree => new branch

            if nb_nodes_in_tree == 1:
                branches.append([n1] + path)  # initial branch of the tree
            else:
                branches.append(path + [n1])  # path as a new branch

            state[path] = 1  # mark nodes in path as in the tree
            nb_nodes_in_tree += len(path)

            # Restart the walk from a random node among those not visited
            nodes_not_visited = np.where(state == -1)[0]
            if nodes_not_visited.size:
                n0 = rng.choice(nodes_not_visited)  # size=1)[0]
                path = [n0]

    tree_edges = list(chain.from_iterable(map(lambda x: zip(x[:-1], x[1:]),
                                              branches)))
    wilson_tree_graph.add_edges_from(tree_edges)

    return wilson_tree_graph


def ust_sampler_aldous_broder(list_of_neighbors, root=None,
                              random_state=None):

    rng = check_random_state(random_state)

    # Initialize the tree
    aldous_tree_graph = nx.Graph()
    nb_nodes = len(list_of_neighbors)

    # Initialize the root, if root not specified start from any node
    n0 = root if root else rng.choice(nb_nodes)  # size=1)[0]
    visited = np.zeros(nb_nodes, dtype=bool)
    visited[n0] = True
    nb_nodes_in_tree = 1

    tree_edges = np.zeros((nb_nodes - 1, 2), dtype=np.int)

    while nb_nodes_in_tree < nb_nodes:

        # visit a neighbor of n0 uniformly at random
        n1 = rng.choice(list_of_neighbors[n0])  # size=1)[0]

        if visited[n1]:
            pass  # continue the walk
        else:  # create edge (n0, n1) and continue the walk
            tree_edges[nb_nodes_in_tree - 1] = [n0, n1]
            visited[n1] = True  # mark it as in the tree
            nb_nodes_in_tree += 1

        n0 = n1

    aldous_tree_graph.add_edges_from(tree_edges)

    return aldous_tree_graph


def uniform_permutation(N, random_state=None):
    """ Draw a perputation :math:`\\sigma \\in \\mathfrak{S}_N` uniformly at random using Fisher-Yates' algorithm

    .. seealso::

        - `Fisher–Yates_shuffle <https://en.wikipedia.org/wiki/Fisher%E2%80%93Yates_shuffle>_

        - `Numpy shuffle <https://github.com/numpy/numpy/blob/d429f0fe16c0407509b1f20d997bf94f1027f61b/numpy/random/mtrand.pyx#L4027>_`
    """
    rng = check_random_state(random_state)

    sigma = np.arange(N)
    for i in range(N - 1, 0, -1):  # reversed(range(1, N))
        j = rng.randint(0, i + 1)
        if j == i:
            continue
        sigma[j], sigma[i] = sigma[i], sigma[j]

    # for i in range(N - 1):
    #     j = rng.randint(i, N)
    #     sigma[j], sigma[i] = sigma[i], sigma[j]

    return sigma


# def wrapper_plot_descent(func):
#     """ Figure settings, ticks mostly
#     """
#     @functools.wraps(func)
#     def wrapper(*args, **kwargs):

#         ax, size = func(*args, **kwargs)

#         # Spine options
#         ax.spines['bottom'].set_position('center')
#         ax.spines['left'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['right'].set_visible(False)

#         # Ticks options
#         minor_ticks = np.arange(0, size + 1)
#         major_ticks = np.arange(0, size + 1, 10)
#         ax.set_xticks(major_ticks)
#         ax.set_xticks(minor_ticks, minor=True)
#         ax.set_xticklabels(major_ticks, fontsize=15)
#         ax.xaxis.set_ticks_position('bottom')

#         ax.tick_params(
#             axis='y',           # changes apply to the y-axis
#             which='both',       # both major and minor ticks are affected
#             left=False,         # ticks along the left edge are off
#             right=False,        # ticks along the right edge are off
#             labelleft=False)    # labels along the left edge are off

#         ax.xaxis.grid(True)
#         ax.set_xlim([-1, size + 1])
#         ax.legend(bbox_to_anchor=(0, 0.85),
#                   frameon=False,
#                   prop={'size': 15})

#     return wrapper


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
    """ Compute the xy coordinates of the boxes defining the young diagram, using the russian convention.

    :param young_diag:
        points
    :type  young_diag:
        array_like

    :return:
        :math:`\\omega(x)`
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
        [np.hstack([x_hor, x_vert]), np.hstack([y_hor, y_vert])])

    rot_45_and_scale = np.array([[1.0, -1.0],
                                 [1.0, 1.0]])

    return xy_young_fr.dot(rot_45_and_scale.T)


def limit_shape(x):
    """ Evaluate :math:`\\omega(x)` the limit-shape function :cite:`Ker96`

    .. math::

        \\omega(x) =
        \\begin{cases}
            |x|, &\\text{if } |x|\\geq 2\\
            \\frac{2}{\\pi} \\left(x \\arcsin\\left(\\frac{x}{2}\\right) + \\sqrt{4-x^2} \\right) &\\text{otherwise } \\end{cases}

    :param x:
        points
    :type x:
        array_like

    :return:
        :math:`\\omega(x)`
    :rtype:
        array_like

    .. seealso::

        - :func:`plot_diagram <plot_diagram>`
        - :cite:`Ker96`
    """

    w_x = np.zeros_like(x)

    abs_x_gt2 = np.abs(x) >= 2.0

    w_x[abs_x_gt2] = np.abs(x[abs_x_gt2])
    w_x[~abs_x_gt2] = x[~abs_x_gt2] * np.arcsin(0.5 * x[~abs_x_gt2])\
                      + np.sqrt(4.0 - x[~abs_x_gt2]**2)
    w_x[~abs_x_gt2] *= 2.0 / np.pi

    return w_x
