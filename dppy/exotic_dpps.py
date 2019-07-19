# coding: utf8
""" Implementation of exotic DPP objects:

- Uniform spanning trees :class:`UST`
- Descent procresses :class:`Descent`:

    * :class:`CarriesProcess`
    * :class:`DescentProcess`
    * :class:`VirtualDescentProcess`

- :class:`PoissonizedPlancherel` measure

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/exotic_dpps/index.html>`_
"""

import abc

from sys import platform as _platform
# https://stackoverflow.com/questions/1854/python-what-os-am-i-running-on
if _platform.startswith('linux'):
    # linux
    pass
elif _platform == "darwin":
    # MAC OS X
    # https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/
    # import matplotlib
    # matplotlib.use('TkAgg')
    pass

import matplotlib.pyplot as plt
from matplotlib import collections as mc  # see plot_diagram

import numpy as np

from scipy.linalg import qr

# For Uniform Spanning Trees
import networkx as nx
from dppy.exotic_dpps_core import ust_sampler_wilson, ust_sampler_aldous_broder
from dppy.exact_sampling import proj_dpp_sampler_eig

# For DescentProcess
import re  # to convert class names to string in
# from dppy.exotic_dpps_core import wrapper_plot_descent

# For Poissonized Plancherel measure
from dppy.exotic_dpps_core import RSK, xy_young_ru, limit_shape

# For both Descent Processes and Poissonized Plancherel
from dppy.exotic_dpps_core import uniform_permutation

from dppy.utils import check_random_state


#####################
# Descent Processes #
#####################
class Descent(metaclass=abc.ABCMeta):

    def __init__(self):

        self.name =\
            ' '.join(re.findall('[A-Z][^A-Z]*', self.__class__.__name__))
        self.list_of_samples = []
        self.size = 100

    @property
    @abc.abstractmethod
    def _bernoulli_param(self):
        """Parameter of the corresponding process formed by i.i.d. Bernoulli variables.
        This parameter corresponds to the probability that a descent occurs any index"""

        return 0.5

    @abc.abstractmethod
    def sample(self, random_state=None):
        """Sample from corresponding process"""

    def flush_samples(self):
        """ Empty the :py:attr:`list_of_samples` attribute.
        """
        self.list_of_samples = []

    def plot(self, vs_bernoullis=True, random_state=None):
        """Display the last realization of the process.
        If ``vs_bernoullis=True`` compare it to a sequence of i.i.d. Bernoullis with parameter ``_bernoulli_param``

        .. seealso::

            - :py:meth:`sample`
        """
        rng = check_random_state(random_state)
        title = 'Realization of the {} process'.format(self.name)

        fig, ax = plt.subplots(figsize=(19, 2))

        sampl = self.list_of_samples[-1]
        ax.scatter(sampl,
                   np.zeros_like(sampl) + (1.0 if vs_bernoullis else 0.0),
                   color='b', s=20, label=self.name)

        if vs_bernoullis:
            title += r' vs independent Bernoulli variables with parameter $p$={:.3f}'.format(self._bernoulli_param)

            bern = np.where(rng.rand(self.size) < self._bernoulli_param)[0]
            ax.scatter(bern, -np.ones_like(bern),
                       color='r', s=20, label='Bernoullis')

        plt.title(title)

        # Spine options
        ax.spines['bottom'].set_position('center')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Ticks options
        minor_ticks = np.arange(0, self.size + 1)
        major_ticks = np.arange(0, self.size + 1, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticklabels(major_ticks, fontsize=15)
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(
            axis='y',           # changes apply to the y-axis
            which='both',       # both major and minor ticks are affected
            left=False,         # ticks along the left edge are off
            right=False,        # ticks along the right edge are off
            labelleft=False)    # labels along the left edge are off

        ax.xaxis.grid(True)
        ax.set_xlim([-1, self.size + 1])
        ax.legend(bbox_to_anchor=(0, 0.85),
                  frameon=False,
                  prop={'size': 15})


class CarriesProcess(Descent):
    """ DPP on :math:`\\{1, \\dots, N-1\\}` (with a non symmetric kernel) derived from the cumulative sum of :math:`N` i.i.d. digits in :math:`\\{0, \\dots, b-1\\}`.

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

        str_info = ['Carries process in base {}'.format(self.base),
                    'Number of samples = {}'.format(len(self.list_of_samples))]

        return '\n'.join(str_info)

    @property
    def _bernoulli_param(self):
        return 0.5 * (1 - 1 / self.base)

    def sample(self, size=100, random_state=None):
        """ Compute the cumulative sum (in base :math:`b`) of a sequence of i.i.d. digits and record the position of carries.

        :param size:
            size of the sequence of i.i.d. digits in :math:`\\{0, \\dots, b-1\\}`

        :type size:
            int
        """
        rng = check_random_state(random_state)

        self.size = size
        A = rng.randint(0, self.base, self.size)
        B = np.mod(np.cumsum(A), self.base)

        carries = 1 + np.where(B[:-1] > B[1:])[0]

        self.list_of_samples.append(carries.tolist())


class DescentProcess(Descent):
    """ DPP on :math:`\\{1, \\dots, N-1\\}` associated to the descent process on the symmetric group :math:`\\mathfrak{S}_N`.

        .. seealso::

            - :cite:`BoDiFu10`
            - :ref:`descent_process`
    """

    def __init__(self):
        super().__init__()

    def __str__(self):

        str_info = ['Descent process',
                    'Number of samples = {}'.format(len(self.list_of_samples))]

        return '\n'.join(str_info)

    @property
    def _bernoulli_param(self):
        return 0.5

    def sample(self, size=100, random_state=None):
        """ Draw a permutation :math:`\\sigma \\in \\mathfrak{S}_N` uniformly at random and record the descents i.e. :math:`\\{ i ~;~ \\sigma_i > \\sigma_{i+1} \\}`.

        :param size:
            size of the permutation i.e. degree :math:`N` of :math:`\\mathfrak{S}_N`.

        :type size:
            int
        """

        rng = check_random_state(random_state)

        self.size = size
        sigma = uniform_permutation(self.size,
                                    random_state=rng)

        descent = 1 + np.where(sigma[:-1] > sigma[1:])[0]

        self.list_of_samples.append(descent.tolist())


class VirtualDescentProcess(Descent):
    """ This is a DPP on :math:`\\{1, \\dots, N-1\\}` with a non symmetric kernel appearing in (or as a limit of) the descent process on the symmetric group :math:`\\mathfrak{S}_N`.

    .. seealso::

        - :cite:`Kam18`
        - :ref:`limiting_descent_process`
        - :class:`DescentProcess`
    """

    def __init__(self, x_0=0.5):

        super().__init__()
        if not (0 <= x_0 <= 1):
            raise ValueError("x_0 must be in [0,1]")
        self.x_0 = x_0

    def __str__(self):

        str_info = ['Limitting Descent process for vitural permutations',
                    'Number of samples = {}'.format(len(self.list_of_samples))]

        return '\n'.join(str_info)

    @property
    def _bernoulli_param(self):
        return 0.5 * (1 - self.x_0**2)

    def sample(self, size=100, random_state=None):
        """ Draw a permutation uniformly at random and record the descents i.e. indices where :math:`\\sigma(i+1) < \\sigma(i)` and something else...

        :param size:
            size of the permutation i.e. degree :math:`N` of :math:`\\mathfrak{S}_N`.

        :type size:
            int

        .. seealso::

            - :cite:`Kam18`, Sec ??

        .. todo::

            ask @kammmoun to complete the docsting and Section in see also
        """

        rng = check_random_state(random_state)

        self.size = size
        sigma = uniform_permutation(self.size + 1,
                                    random_state=rng)

        X = sigma[:-1] > sigma[1:]  # Record the descents in permutation

        Y = rng.binomial(n=2, p=self.x_0, size=self.size + 1) != 1

        descent = [i for i in range(self.size)
                   if (~Y[i] and Y[i + 1])
                   or (~Y[i] and ~Y[i + 1] and X[i])]

        self.list_of_samples.append(descent)


##########################
# Poissonized Plancherel #
##########################
class PoissonizedPlancherel:
    """ DPP on partitions associated to the Poissonized Plancherel measure

    :param theta:
        Poisson parameter i.e. expected length of permutation

    :type theta:
        int, default 10

    .. seealso::

        - :cite:`Bor09` Section 6
        - :ref:`poissonized_plancherel_measure`
    """

    def __init__(self, theta=10):

        self.theta = theta  # Poisson param = expected length of permutation
        self.list_of_young_diag = []
        self.list_of_samples = []

    def __str__(self):

        str_info = ['Poissonized Plancherel measure\
                    with parameter {}'.format(self.theta),
                    'Number of samples = {}'.format(len(self.list_of_samples))]

        return '\n'.join(str_info)

    # def info(self):
    #     """ Print infos about the :class:`UST` object
    #     """
    #     print(self.__str__())

    def sample(self, random_state=None):
        """ Sample from the Poissonized Plancherel measure.

        :param random_state:
        :type random_state:
            None, np.random, int, np.random.RandomState
        """
        rng = check_random_state(random_state)

        N = rng.poisson(self.theta)
        sigma = uniform_permutation(N, random_state=rng)
        P, _ = RSK(sigma)

        # young_diag = [len(row) for row in P]
        young_diag = np.fromiter(map(len, P), dtype=int)
        self.list_of_young_diag.append(young_diag)
        # sampl = [len(row) - i + 0.5 for i, row in enumerate(P, start=1)]
        sampl = young_diag - np.arange(0.5, young_diag.size)
        self.list_of_samples.append(sampl.tolist())

    def plot(self, title=''):
        """Display the process on the real line

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :py:meth:`sample`
        """

        sampl = self.list_of_samples[-1]

        # Display the reparametrized Plancherel sample
        fig, ax = plt.subplots(figsize=(19, 2))

        ax.scatter(sampl, np.zeros_like(sampl), color='blue', s=20)

        # Spine options
        ax.spines['bottom'].set_position('center')
        ax.spines['left'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Ticks options
        x_max = np.max(np.abs(sampl)) + 0.5
        minor_ticks = np.arange(-x_max, x_max + 1)
        major_ticks = np.arange(-100, 100 + 1, 10)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)
        ax.set_xticklabels(major_ticks, fontsize=15)
        ax.xaxis.set_ticks_position('bottom')

        ax.tick_params(
            axis='y',           # changes apply to the y-axis
            which='both',       # both major and minor ticks are affected
            left=False,         # ticks along the left edge are off
            right=False,        # ticks along the right edge are off
            labelleft=False)    # labels along the left edge are off

        ax.xaxis.grid(True)
        ax.set_xlim([-x_max - 2, x_max + 2])
        # ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

        str_title = r'Realization of the DPP associated to the Poissonized Plancherel measure with parameter $\theta=${}'.format(self.theta)
        plt.title(title if title else str_title)

    def plot_diagram(self, normalization=False):
        """ Display the Young diagram (russian convention), the associated sample and potentially rescale the two to visualize the limit-shape theorem :cite:`Ker96`.
        The sample corresponds to the projection onto the real line of the descending surface edges.

        :param normalization:
            If ``normalization=True``, the Young diagram and the corresponding sample are scaled by a factor :math:`\\sqrt{\\theta}` and the limiting

        :type normalization:
            bool, default False

        .. seealso::

            - :py:meth:`sample`
            - :py:meth:`plot`
            - :cite:`Ker96`
        """

        y_diag = self.list_of_young_diag[-1]
        sampl = self.list_of_samples[-1].copy()

        x_max = 1.1 * max(y_diag.size, y_diag[0])
        xy_young = xy_young_ru(y_diag)

        if normalization:
            sampl /= np.sqrt(self.theta)
            x_max /= np.sqrt(self.theta)
            xy_young /= np.sqrt(self.theta)

        fig, ax = plt.subplots(figsize=(12, 8))

        # Display corresponding sample
        ax.scatter(sampl, np.zeros_like(sampl),
                   s=3, label='sample')
        # Display absolute value wedge
        ax.plot([-x_max, 0.0, x_max], [x_max, 0.0, x_max],
                c='k', lw=1)
        # Display young diagram in russian notation
        lc = mc.LineCollection(xy_young.reshape((-1, 2, 2)),
                               color='k', linewidths=2)
        ax.add_collection(lc)
        # Display limit shape
        if normalization:
            x_lim_sh = np.linspace(-x_max, x_max, 100)
            ax.plot(x_lim_sh, limit_shape(x_lim_sh),
                    c='r', label='limit shape')

        # Display stems linking sample on real line and descent in young diag
        # xy_y_diag = np.column_stack([y_diag,
        # np.arange(0.5, y_diag.size)]).dot(rot_45_and_scale.T)
        # if normalization:
        #     xy_y_diag /= np.sqrt(theta)
        # plt.scatter(xy_y_diag[:,0], np.zeros_like(y_diag), color='r')
        # plt.stem(xy_y_diag[:,0], xy_y_diag[:,1], linefmt='C0--', basefmt=' ')

        plt.legend(loc='best')
        plt.axis('equal')

        str_title = r'Young diagram associated to Poissonized Plancherel measure with parameter $\theta=${}'.format(self.theta)
        plt.title(str_title)


##########################
# Uniform Spanning Trees #
##########################
class UST:
    """ DPP on edges of a connected graph :math:`G` with correlation kernel the projection kernel onto the span of the rows of the incidence matrix :math:`\\text{Inc}` of :math:`G`.

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

        if nx.is_connected(graph):
            self.graph = graph
        else:
            raise ValueError('graph not connected')

        self.nodes = list(self.graph.nodes())
        self.nb_nodes = self.graph.number_of_nodes()  # len(self.graph)

        self.edges = list(self.graph.edges())
        self.nb_edges = self.graph.number_of_edges()

        self.neighbors = [list(graph.neighbors(v))
                          for v in range(self.nb_nodes)]

        self.sampling_mode = 'Wilson'  # 'Aldous-Broder', 'DPP_exact'
        self.list_of_samples = []

        self.kernel = None
        self.kernel_eig_vecs = None

    def __str__(self):

        str_info = ['Uniform Spanning Tree measure on a graph with:',
                    '- {} nodes'.format(self.nb_nodes),
                    '- {} edges'.format(self.nb_edges),
                    'Sampling mode = {}'.format(self.sampling_mode),
                    'Number of samples = {}'.format(len(self.list_of_samples))]

        return '\n'.join(str_info)

    # def info(self):
    #     """ Print infos about the :class:`UST` object
    #     """
    #     print(self.__str__())

    def flush_samples(self):
        """ Empty the :py:attr:`list_of_samples` attribute.
        """
        self.list_of_samples = []

    def sample(self, mode='Wilson', root=None, random_state=None):
        """ Sample a spanning of the underlying graph uniformly at random.
        It generates a networkx graph object.

        :param mode:

            - ``'Wilson'``
            - ``'Aldous-Broder'``
            - ``'DPP_exact'``

        :type mode:
            string, default ``'Wilson'``

        :param root:
            Starting node of the random walk
        :type root:
            int

        :param random_state:
        :type random_state:
            None, np.random, int, np.random.RandomState

        .. seealso::

            - Wilson algorithm :cite:`PrWi98`
            - Aldous-Broder :cite:`Ald90`
        """

        rng = check_random_state(random_state)

        self.sampling_mode = mode

        if self.sampling_mode == 'Wilson':
            sampl = ust_sampler_wilson(self.neighbors,
                                       random_state=rng)

        elif self.sampling_mode == 'Aldous-Broder':
            sampl = ust_sampler_aldous_broder(self.neighbors,
                                              random_state=rng)

        elif self.sampling_mode == 'DPP_exact':

            if self.kernel_eig_vecs is None:
                self.__compute_kernel_eig_vecs()

            dpp_sample = proj_dpp_sampler_eig(self.kernel_eig_vecs,
                                              random_state=rng)

            sampl = nx.Graph()
            edges_finite_dpp = [self.edges[e] for e in dpp_sample]
            sampl.add_edges_from(edges_finite_dpp)

        else:
            err_print = ('Invalid sampling mode',
                         'Choose: `Wilson`, `Aldous-Broder`, `DPP_exact`',
                         'Given {}'.format(mode))
            raise ValueError()

        self.list_of_samples.append(sampl)

    def compute_kernel(self):
        """ Compute the orthogonal projection kernel :math:`\\mathbf{K} = \\text{Inc}^+ \\text{Inc}` i.e. onto the span of the rows of the vertex-edge incidence matrix :math:`\\text{Inc}` of size :math:`|V| \\times |E|`.

        In fact, for a connected graph, :math:`\\text{Inc}` has rank :math:`|V|-1` and any row can be discarded to get an basis of row space. If we note :math:`A` the amputated version of :math:`\\text{Inc}`, then :math:`\\text{Inc}^+ = A^{\\top}[AA^{\\top}]^{-1}`.

        In practice, we orthogonalize the rows of :math:`A` to get the eigenvectors :math:`U` of :math:`\\mathbf{K}=UU^{\\top}`.

        .. seealso::

            - :py:meth:`plot_kernel`
        """

        if self.kernel is None:
            if self.kernel_eig_vecs is None:
                self.__compute_kernel_eig_vecs()  # QR(Inc[:-1,:])
            # K = UU.T
            self.kernel = self.kernel_eig_vecs.dot(self.kernel_eig_vecs.T)

    def __compute_kernel_eig_vecs(self):
        """ See explaination in :func:`compute_kernel <compute_kernel>`
        """

        inc_mat = nx.incidence_matrix(self.graph, oriented=True)
        # Discard any row e.g. the last one
        A = inc_mat[:-1, :].toarray()
        # Orthonormalize rows of A
        self.kernel_eig_vecs, _ = qr(A.T, mode='economic')

    def plot(self, title=''):
        """ Display the last realization (spanning tree) of the corresponding :class:`UST` object.

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :py:meth:`sample`
        """

        graph_to_plot = self.list_of_samples[-1]

        plt.figure(figsize=(4, 4))

        pos = nx.circular_layout(graph_to_plot)
        nx.draw_networkx(graph_to_plot,
                         pos=pos,
                         node_color='orange',
                         with_labels=True)
        plt.axis('off')

        str_title = 'UST with {} algorithm'.format(self.sampling_mode)
        plt.title(title if title else str_title)


    def plot_graph(self, title=''):
        """Display the original graph defining the :class:`UST` object

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :func:`compute_kernel <compute_kernel>`
        """

        edge_lab = [r'$e_{}$'.format(i) for i in range(self.nb_edges)]
        edge_labels = dict(zip(self.edges, edge_lab))
        node_labels = dict(zip(self.nodes, self.nodes))

        plt.figure(figsize=(4, 4))

        pos = nx.circular_layout(self.graph)
        nx.draw_networkx(self.graph,
                         pos=pos,
                         node_color='orange',
                         with_labels=True,
                         width=3)
        nx.draw_networkx_labels(self.graph,
                                pos,
                                node_labels)
        nx.draw_networkx_edge_labels(self.graph,
                                     pos,
                                     edge_labels,
                                     font_size=20)

        plt.axis('off')

        str_title = 'Original graph'
        plt.title(title if title else str_title)

    def plot_kernel(self, title=''):
        """Display a heatmap of the underlying orthogonal projection kernel :math:`\\mathbf{K}` associated to the DPP underlying the :class:`UST` object

        :param title:
            Plot title

        :type title:
            string

        .. seealso::

            - :func:`compute_kernel <compute_kernel>`
        """

        self.compute_kernel()

        fig, ax = plt.subplots(1, 1)

        heatmap = ax.pcolor(self.kernel, cmap='jet')

        ax.set_aspect('equal')

        ticks = np.arange(self.nb_edges)
        ticks_label = [r'${}$'.format(tic) for tic in ticks]

        ax.xaxis.tick_top()
        ax.set_xticks(ticks + 0.5, minor=False)

        ax.invert_yaxis()
        ax.set_yticks(ticks + 0.5, minor=False)

        ax.set_xticklabels(ticks_label, minor=False)
        ax.set_yticklabels(ticks_label, minor=False)

        str_title = 'UST kernel i.e. transfer current matrix'
        plt.title(title if title else str_title, y=1.08)

        plt.colorbar(heatmap)
