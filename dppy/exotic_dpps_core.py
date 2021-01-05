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

import numpy as np
from bisect import bisect_right  # for RSK

from dppy.utils import check_random_state


def ust_sampler_wilson(g, root=None, random_state=None):
    """Generate a uniform spanning tree of g at root using [Wilson's algorithm](https://dl.acm.org/doi/10.1145/237814.237880).

    :param g: Connected graph
    :type g: nx.Graph

    :param root: Any node of g, defaults to None. If None, the root is chosen uniformly at random among g.nodes.
    :type root: list, optional

    :return: uniform spanning tree of g
    :rtype: nx.Graph
    """

    try:
        import networkx as nx
    except ImportError:
        raise ValueError('The networkx package is required to sample spanning trees (see setup.py).')

    rng = check_random_state(random_state)

    if root is None:
        nodes = list(g.nodes)
        root = nodes[rng.randint(len(nodes))]
    elif root not in g:
        raise ValueError("root not in g.nodes")

    tree = {root}
    successor = dict.fromkeys(g.nodes, None)
    del successor[root]

    for i in g.nodes:
        # Run a natural random walk from i until it hits a node in tree
        u = i
        while u not in tree:
            neighbors = list(g.neighbors(u))
            successor[u] = neighbors[rng.randint(len(neighbors))]
            u = successor[u]

        # Record Erase first loop created during the random walk
        u = i
        while u not in tree:
            tree.add(u)
            u = successor[u]

    return nx.from_edgelist(successor.items())


def ust_sampler_aldous_broder(g, root=None, random_state=None):
    """Generate a uniform spanning tree of g at root using [Aldous](https://epubs.siam.org/doi/10.1137/0403039)-[Broder](https://doi.org/10.1109/SFCS.1989.63516)'s algorithm

    :param g: Connected graph
    :type g: nx.Graph

    :param root: Any node of g, defaults to None. If None, the root is chosen uniformly at random among g.nodes.
    :type root: list, optional

    :return: uniform spanning tree of g
    :rtype: nx.Graph
    """
    try:
        import networkx as nx
    except ImportError:
        raise ValueError('The networkx package is required to sample spanning trees (see setup.py).')
    rng = check_random_state(random_state)

    if root is None:
        nodes = list(g.nodes)
        root = nodes[rng.randint(len(nodes))]
    elif root not in g:
        raise ValueError("root not in g.nodes")

    # Run a natural random walk from root a until all nodes are visited
    tree = {root: None}
    u = root
    while len(tree) < g.number_of_nodes():
        neighbors = list(g.neighbors(u))
        v = neighbors[rng.randint(len(neighbors))]
        # Record an edge when reaching an unvisited node
        if v not in tree:
            tree[v] = u
        u = v

    del tree[root]

    return nx.from_edgelist(tree.items())


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
