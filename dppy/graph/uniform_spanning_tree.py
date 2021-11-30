import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from dppy.finite.dpp import FiniteDPP
from dppy.graph.uniform_spanning_tree_samplers import (
    ust_sampler_aldous_broder,
    ust_sampler_wilson,
)
from dppy.utils import check_random_state


class UST:
    r"""DPP on edges of a connected graph :math:`G` with correlation kernel the projection kernel onto the span of the rows of the incidence matrix :math:`\text{Inc}` of :math:`G`.

    This DPP corresponds to the uniform measure on spanning trees (UST) of :math:`G`.

    :param graph:
        Connected undirected graph
    :type graph:
        networkx graph

    .. seealso::

        - :ref:`UST`
        - :ref:`Definition of DPP <finite_dpps_definition>`
    """

    def __init__(self, graph):
        assert nx.is_connected(graph), "graph must be connected"
        self.graph = graph

        inc_mat = nx.incidence_matrix(graph, oriented=True)
        A = inc_mat[:-1, :].toarray()
        self._dpp = FiniteDPP(
            kernel_type="correlation",
            projection=True,
            hermitian=True,
            A_zono=A,
        )

        self.sampling_method = "Wilson"  # Default (avoid eig_vecs computation)
        self.list_of_samples = []

    def __str__(self):

        str_info = [
            "Uniform Spanning Tree measure on a graph with:",
            "- {} nodes".format(self.graph.number_of_nodes()),
            "- {} edges".format(self.graph.number_of_edges()),
            "Sampling method = {}".format(self.sampling_method),
            "Number of samples = {}".format(len(self.list_of_samples)),
        ]

        return "\n".join(str_info)

    @property
    def kernel(self):
        r"""Compute the orthogonal projection kernel :math:`\mathbf{K} = \text{Inc}^+ \text{Inc}` i.e. onto the span of the rows of the vertex-edge incidence matrix :math:`\text{Inc}` of size :math:`|V| \times |E|`.

        For a connected graph, :math:`\text{Inc}` has rank :math:`|V|-1` and any row can be discarded to get a basis of row space. If we note :math:`A` the amputated version of :math:`\text{Inc}`, then :math:`\text{Inc}^+ = A^{\top}[AA^{\top}]^{-1}`.

        In practice, we orthogonalize the rows of :math:`A` to get the eigenvectors :math:`U` of :math:`\mathbf{K}=UU^{\top}`.

        .. seealso::

            - :py:meth:`plot_kernel`
            - :py:meth:`compute_kernel_eig_vecs`
        """
        self._dpp.compute_K()
        return self._dpp.K

    def flush_samples(self):
        """Empty the :py:attr:`list_of_samples` attribute."""
        self.list_of_samples = []

    def sample(self, method="wilson", random_state=None, **params):
        """Sample a spanning tree of the underlying graph, uniformly at random.

        :param method:
            - Markov-chain-based samplers: ``'wilson'`` :cite:`PrWi98`, or ``'aldous-broder'`` :cite:`Ald90`,
            - DPP-based samplers: see :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_exact`, default is ``"spectral"``.
        :type method:
            string, default ``'Wilson'``

        :param random_state:
        :type random_state:
            None, np.random, int, np.random.RandomState

        :return:
            Uniform spanning tree.
        :rtype:
            networkx.Graph
        """
        rng = check_random_state(random_state)

        _method = method.lower()
        markov_chain_samplers = {
            "wilson": ust_sampler_wilson,
            "aldous-broder": ust_sampler_aldous_broder,
        }
        if _method in markov_chain_samplers:
            sampler = markov_chain_samplers[_method]
            sample = sampler(self.graph, random_state=rng)
        else:
            dpp_sample = self._dpp.sample_exact(
                method=_method, random_state=rng, **params
            )
            sample = self._dpp_sample_to_nx_graph(dpp_sample)

        self.sampling_method = method
        self.list_of_samples.append(sample)
        return sample

    def _dpp_sample_to_nx_graph(self, dpp_sample):
        edges = list(self.graph.edges)
        return nx.from_edgelist([edges[i] for i in dpp_sample])

    def plot(self, sample, ax=None):
        """Display the last realization (spanning tree) of the corresponding :class:`UST` object.

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :py:meth:`sample`
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))

        plt.title(
            "Uniform spanning tree generated with {} method".format(
                self.sampling_method
            )
        )

        pos = nx.circular_layout(self.graph)
        nx.draw_networkx(
            sample,
            pos=pos,
            node_color="orange",
            with_labels=True,
            width=3,
            ax=ax,
        )

        labs = {e: r"$e_{}$".format(i) for i, e in enumerate(self.graph.edges)}
        edge_labs = {e: labs[e if e in labs else e[::-1]] for e in sample.edges}
        nx.draw_networkx_edge_labels(
            sample,
            pos=pos,
            edge_labels=edge_labs,
            font_size=20,
            ax=ax,
        )

        plt.axis("off")
        return ax

    def plot_graph(self, ax=None):
        """Display the original graph defining the :class:`UST` object

        :param title:
            Plot title

        :type title:
            string
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        plt.title("Original graph")

        pos = nx.circular_layout(self.graph)
        nx.draw_networkx(
            self.graph, pos=pos, node_color="orange", with_labels=True, width=3, ax=ax
        )
        # nx.draw_networkx_labels(self.graph,
        #                         pos,
        #                         node_labels)
        labs = {e: r"$e_{}$".format(i) for i, e in enumerate(self.graph.edges)}
        nx.draw_networkx_edge_labels(
            self.graph, pos=pos, edge_labels=labs, font_size=20, ax=ax
        )

        plt.axis("off")
        return ax

    def plot_kernel(self, ax=None):
        r"""Display a heatmap of the underlying orthogonal projection kernel :math:`\mathbf{K}` associated to the DPP underlying the :class:`UST` object

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :func:`compute_kernel <compute_kernel>`
        """

        if ax is None:
            fig, ax = plt.subplots(figsize=(4, 4))
        else:
            fig = ax.get_figure()

        plt.title("Correlation K kernel: transfer current matrix", y=1.08)

        heatmap = ax.pcolor(self.kernel, cmap="jet")

        ax.set_aspect("equal")

        ticks = np.arange(self.graph.number_of_edges())
        ticks_label = [r"${}$".format(tic) for tic in ticks]

        ax.xaxis.tick_top()
        ax.set_xticks(ticks + 0.5, minor=False)

        ax.invert_yaxis()
        ax.set_yticks(ticks + 0.5, minor=False)

        ax.set_xticklabels(ticks_label, minor=False)
        ax.set_yticklabels(ticks_label, minor=False)

        # Adapt size of colbar to plot
        # https://stackoverflow.com/questions/18195758/set-matplotlib-colorbar-size-to-match-graph
        cax = fig.add_axes(
            [
                ax.get_position().x1 + 0.02,
                ax.get_position().y0,
                0.05,
                ax.get_position().height,
            ]
        )
        plt.colorbar(heatmap, cax=cax)
        return ax
