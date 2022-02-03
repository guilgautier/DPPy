# coding: utf8
""" Unit tests:

- :class:`TestUniformityUniformSpanningTreeSampler` to check that the different procedures used for uniform spanning trees of a graph actually sample spanning trees uniformly at random.
"""
import itertools as itt
import unittest
from collections import Counter

import numpy as np
from networkx import erdos_renyi_graph, incidence_matrix, is_connected
from scipy.stats import chisquare

from dppy.graph.uniform_spanning_tree import UST
from dppy.utils import det_ST


class UniformityOfSamplerForUniformSpanningTree(unittest.TestCase):
    """Test uniformity the sampling procedures for sampling spanning trees of a graph uniformly at random on an Erdos-Renyi graph :math:`G(n,p)`"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Sample a connected Erdos-Renyi graph
        nv, p = 5, 0.4
        nb_st_min, nb_st_max = 5, 10

        it_max = 100
        for _ in range(it_max):
            g = erdos_renyi_graph(nv, p)
            if not is_connected(g):
                continue
            ne = g.number_of_edges()
            A = incidence_matrix(g, oriented=True)[:-1, :].toarray()
            candidate = itt.combinations(range(ne), nv - 1)
            list_st = [st for st in candidate if det_ST(A, range(nv - 1), st)]
            if nb_st_min <= len(list_st) <= nb_st_max:
                break
        else:
            raise ValueError("No satisfactory Erdos-Renyi graph found")

        self.nb_spanning_trees = len(list_st)
        self.ust = UST(g)
        self.nb_samples = 1000

    def test_projection_kernel_computation(self):
        """UST is a DPP associated to the projection kernel onto the row span of the vertex-edge-incidence matrix"""
        inc = incidence_matrix(self.ust.graph, oriented=True).todense()
        expected_kernel = np.linalg.pinv(inc).dot(inc)
        self.assertTrue(np.allclose(self.ust.kernel, expected_kernel))

    @staticmethod
    def sample_to_label(graph):
        """Join egdes of a sample to from the ID of the corresponding spanning tree
        Ex:
        [(3, 2), (2, 0), (2, 1), (0, 4)] -> '[0, 2][0, 4][1, 2][2, 3]'
        """
        return "".join(map(str, sorted(map(sorted, graph.edges()))))

    def test_uniformity_adequation(self):

        samplers = (
            "wilson",
            "aldous-broder",
            "spectral",
            "schur",
            "chol",
            "sequential",
        )

        for method in samplers:
            with self.subTest(method=method):
                self.ust.flush_samples()
                for _ in range(self.nb_samples):
                    self.ust.sample(method=method)
                self.assertTrue(self.uniformity_adequation())

    def uniformity_adequation(self, tol=0.05):
        """Perform chi-square test to check that the different spanning trees sampled have a uniform distribution"""
        counter = Counter(map(self.sample_to_label, self.ust.list_of_samples))
        freq = np.array(list(counter.values())) / self.nb_samples
        theo = np.ones(self.nb_spanning_trees) / self.nb_spanning_trees
        _, pval = chisquare(f_obs=freq, f_exp=theo)
        return pval > tol
