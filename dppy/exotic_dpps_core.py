# coding: utf8
""" Core functions for

- Uniform spanning trees

    * :func:`ust_sampler_wilson`
    * :func:`ust_sampler_aldous_broder`:
"""


from itertools import chain  # create graph edges from path

import numpy as np

from dppy.utils import check_random_state


def ust_sampler_wilson(list_of_neighbors, root=None, random_state=None):
    try:
        import networkx as nx
    except ImportError:
        raise ValueError(
            "The networkx package is required to sample spanning trees (see setup.py)."
        )

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
            nodes_loop = path[knot + 1 :]  # identify nodes forming the loop
            del path[knot + 1 :]  # erase the loop
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

    tree_edges = list(chain.from_iterable(map(lambda x: zip(x[:-1], x[1:]), branches)))
    wilson_tree_graph.add_edges_from(tree_edges)

    return wilson_tree_graph


def ust_sampler_aldous_broder(list_of_neighbors, root=None, random_state=None):

    try:
        import networkx as nx
    except ImportError:
        raise ValueError(
            "The networkx package is required to sample spanning trees (see setup.py)."
        )

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
