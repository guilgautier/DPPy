# coding: utf8
""" Core functions for

- Uniform spanning trees

    * :func:`ust_sampler_wilson`
    * :func:`ust_sampler_aldous_broder`:
"""


import networkx as nx

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
