# coding: utf-8

import numpy as np
import scipy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain
from bisect import bisect_right

try: # Local import
	from .exact_sampling import proj_dpp_sampler_eig_GS
except (SystemError, ImportError):
	from exact_sampling import proj_dpp_sampler_eig_GS

##############################
### Uniform Spanning trees ###
##############################

class UST:
	""" Uniform Spanning Tree object parametrized by

	:param graph: 
		Connected undirected graph

	:type graph:
		networkx graph

	.. seealso::
		
		- :ref:`finite_dpps_definition`
		- :ref:`UST`
	"""

	def __init__(self, graph):
		
		self.graph = graph

		self.nodes = list(self.graph.nodes())
		self.nb_nodes = len(self.graph)

		self.edges = list(self.graph.edges())
		self.nb_edges = self.graph.number_of_edges()

		self.neighbors = [list(graph.neighbors(v)) for v in range(self.nb_nodes)]
		
		self.mode = 'Wilson' # sampling mode
		self.list_of_samples = []

		self.kernel = None
		self.kernel_eig_vecs = None

		# self.list_ST_edge_labels = None
		#degrees = [g.degree(v) for v in nodesices]
		#transition_proba = [[1./deg]*deg for deg in degrees]

	def __str__(self):

		str_info = ['Uniform Spanning Tree measure on a graph with:',
									'- {} nodes'.format(self.nb_nodes),
									'- {} edges'.format(self.nb_edges),
								'Sampling mode = {}'.format(self.mode),
								'Number of samples = {}'.format(len(self.list_of_samples))]

		return '\n'.join(str_info)

	def info(self):
		""" Print infos about the :class:`UST` object
		"""
		print(self.__str__())

	def flush_samples(self):
		""" Empty the ``UST.list_of_samples`` attribute.
		"""
		self.list_of_samples = []

	def sample(self, mode='Wilson'):
		""" Sample exactly from Unif :class:`UST <UST>` object by computing the eigenvalues of random matrices.
		Generates a networkx graph object.

		:param mode:

			- ``'Wilson'``
			- ``'Aldous-Broder'``
			- ``'DPP_exact'``

		:type mode:
			string, default ``'Wilson'``

		.. seealso::

			- Wilson algorithm :cite:`PrWi98`
			- Aldous-Broder :cite:`Ald90`
		"""

		self.mode = mode

		if self.mode=='Wilson':
			sampl = self.__wilson()

		elif self.mode=='Aldous-Broder':
			sampl = self.__aldous()

		elif self.mode=='DPP_exact':

			if self.kernel_eig_vecs is None:
				self.__compute_kernel_eig_vecs()

			dpp_sample = proj_dpp_sampler_eig_GS(self.kernel_eig_vecs)

			g_finite_dpp = nx.Graph()
			edges_finite_dpp = [self.edges[ind] for ind in dpp_sample]
			g_finite_dpp.add_edges_from(edges_finite_dpp)

			sampl = g_finite_dpp

		else:
			raise ValueError('In valid `mode` argument. Choose among `Wilson`, `Aldous-Broder` or `DPP_exact`.\nGiven {}'.format(mode))

		self.list_of_samples.append(sampl)
		
	def compute_kernel(self):
		""" Compute the orthogonal projection kernel :math:`\mathbf{K}` onto the row span of the vertex-edge incidence matrix, refering to the transfer current matrix.
		In fact, one can discard any row of the vertex-edge incidence matrix (:math:`A`) to compute :math:`\mathbf{K}=A^{\top}[AA^{\top}]^{-1}A`.
		In practice, we orthogonalize the rows of :math:`A` to get the eigenvectors :math:`U` of :math:`\mathbf{K}` and thus compute :math:`\mathbf{K}=UU^{\top}`.

		.. seealso::

			- :func:`plot_kernel <plot_kernel>`
		"""

		if self.kernel is None:
			vert_edg_inc = nx.incidence_matrix(self.graph, oriented=True)
			A = vert_edg_inc[:-1,:].toarray() # Discard any row e.g. the last one
			self.kernel_eig_vecs, _ = la.qr(A.T, mode='economic') # Orthog rows of A
			self.kernel = self.kernel_eig_vecs.dot(self.kernel_eig_vecs.T) # K = UU.T
		else:
			pass

	def plot(self, title=''):
		""" Display the last realization (spanning tree) of the corresponding :class:`UST` object.

		:param title:
			Plot title

		:type title:
			string

		.. seealso::

			- :func:`sample <sample>`
		"""

		graph_to_plot = self.list_of_samples[-1]

		fig = plt.figure(figsize=(4,4))
		
		pos=nx.circular_layout(graph_to_plot)
		nx.draw_networkx(graph_to_plot, pos=pos, node_color='orange', 
			with_labels = True)
		plt.axis('off')
		
		str_title = 'UST with {} algorithm'.format(self.mode)
		plt.title(title if title else str_title)

		# plt.savefig('sample_{}_{}.eps'.format(self.mode,len(self.list_of_samples)))

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

		fig = plt.figure(figsize=(4,4))

		pos=nx.circular_layout(self.graph)
		nx.draw_networkx(self.graph, pos=pos, node_color='orange', 
			with_labels = True, width=3)
		nx.draw_networkx_labels(self.graph, pos, node_labels)
		nx.draw_networkx_edge_labels(self.graph, pos, edge_labels, font_size=20)

		plt.axis('off')

		str_title = 'Original graph'
		plt.title(title if title else str_title)
		# plt.savefig('original_graph.eps')


	def plot_kernel(self, title=''):
		"""Display a heatmap of the underlying orthogonal projection kernel :math:`\mathbf{K}` associated to the DPP underlying the :class:`UST` object

		:param title:
			Plot title

		:type title:
			string

		.. seealso::

			- :func:`compute_kernel <compute_kernel>`
		"""

		self.compute_kernel()

		fig, ax = plt.subplots(1,1)

		heatmap = ax.pcolor(self.kernel, cmap='jet')

		ax.set_aspect('equal')

		ticks = np.arange(self.nb_edges)
		ticks_label = [r'${}$'.format(tic) for tic in ticks]

		ax.xaxis.tick_top()
		ax.set_xticks(ticks+0.5, minor=False)

		ax.invert_yaxis()
		ax.set_yticks(ticks+0.5, minor=False)

		ax.set_xticklabels(ticks_label, minor=False)
		ax.set_yticklabels(ticks_label, minor=False)

		str_title = 'UST kernel i.e. transfer current matrix'
		plt.title(title if title else str_title, y=1.08)

		plt.colorbar(heatmap)
		# plt.savefig('kernel.png')

	def __wilson(self, root=None):

		# Initialize the root, if root not specified start from any node
		n0 = root if root else np.random.choice(self.nb_nodes, size=1)[0]
		# -1 = not visited / 0 = in path / 1 = in tree
		nodes_state = -np.ones(self.nb_nodes, dtype=int)
		
		# Initialize the tree
		nodes_state[n0] = 1 # mark root it as in tree
		branches, tree_len = [], 1 # 1 for the root
		path = [] # temporary path
		
		while tree_len < self.nb_nodes:

			# visit a neighbor of n0 uniformly at random
			n1 = np.random.choice(self.neighbors[n0], size=1)[0] 

			if nodes_state[n1] == -1: # not visited => continue the walk

				path.extend([n1]) # add it to the path
				nodes_state[n1] = 0 # mark it as in the path
				n0 = n1 # continue the walk

			if nodes_state[n1] == 0: # loop on the path => erase the loop

				knot = path.index(n1) # find 1st appearence of n1 in the path
				nodes_loop = path[knot+1:] # identify nodes forming the loop
				del path[knot+1:] # erase the loop
				nodes_state[nodes_loop] = -1 # mark loopy nodes as not visited
				n0 = n1 # continue the walk

			elif nodes_state[n1] == 1: # walk hits the tree => new branch of the tree

				if tree_len == 1:
					branches.append([n1]+path) # initial branch of the tree
				else:
					branches.append(path+[n1]) # add the path as a new branch of the tree

				nodes_state[path] = 1 # mark the nodes as in the tree
				tree_len += len(path) # update the length of the tree
				
				# Restart the walk from a random node among those not visited
				nodes_not_visited = np.where(nodes_state==-1)[0]
				if nodes_not_visited.size > 0:
					n0 = np.random.choice(nodes_not_visited, size=1)[0]
					path = [n0]

		wilson_tree_graph = nx.Graph()
		tree_edges = list(chain.from_iterable(
									map(lambda x: zip(x[:-1], x[1:]), branches)))
		wilson_tree_graph.add_edges_from(tree_edges)

		return wilson_tree_graph

	def __aldous(self, root=None):

		# Initialize the root, if root not specified start from any node
		n0 = root if root else np.random.choice(self.nb_nodes, size=1)[0]
	
		# Initialize the tree
		tree_edges, tree_len = [], 1
		visited = np.zeros(self.nb_nodes, dtype=bool)
		visited[n0] = True
		
		while tree_len < self.nb_nodes:

			# visit a neighbor of n0 uniformly at random
			n1 = np.random.choice(self.neighbors[n0], size=1)[0] 

			if visited[n1]: # visited => continue the walk
				pass

			else: # not visited => save the edge (n0, n1) and continue walk
				tree_edges.append((n0, n1))
				visited[n1] = True # mark it as in the tree
				tree_len += 1

			n0 = n1

		aldous_tree_graph = nx.Graph()
		aldous_tree_graph.add_edges_from(tree_edges)

		return aldous_tree_graph
class Descent: 
	def __str__(self):
		pass 
	def info(self):
		""" Print infos about the :class:`UST` object
		"""
		print(self.__str__())
	
	def flush_samples(self):
		""" Empty the ``CarriesProcess.list_of_samples`` attribute.
		"""
		self.list_of_samples = []
	def sample(self, size=100):
		pass
	def plot_vs_bernoullis(self, title='', process_name='',str_title=''):
		"""Display the process on the real line and compare it to a sequence of i.i.d. Bernoullis with parameter :math:`\\frac12(1-\\frac1b)`

		:param title:
			Plot title

		:type title:
			string

		.. seealso::

			- :func:`sample <sample>`
			- :func:`plot <plot>`
		"""

		carries = self.list_of_samples[-1]
		len_car = len(carries)

		ind_tmp = np.random.rand(self.size) < self.bernoulli_param
		bern = np.arange(0, self.size)[ind_tmp]
		len_ber = len(bern)

		# Display Carries and Bernoullis
		fig, ax = plt.subplots(figsize=(19,2))

		ax.scatter(carries, np.ones(len_car), color='b', s=20, label=process_name)
		ax.scatter(bern, -np.ones(len_ber), color='r', s=20, label='Bernoullis')

		# Spine options
		ax.spines['bottom'].set_position('center')
		ax.spines['left'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# Ticks options
		minor_ticks = np.arange(0, self.size+1)                                            
		major_ticks = np.arange(0, self.size+1, 20)                                               
		ax.set_xticks(major_ticks)                                                       
		ax.set_xticks(minor_ticks, minor=True)
		ax.set_xticklabels(major_ticks, fontsize=15)
		ax.xaxis.set_ticks_position('bottom')

		ax.tick_params(
		    axis='y',				# changes apply to the y-axis
		    which='both',		# both major and minor ticks are affected
		    left=False,			# ticks along the left edge are off
		    right=False,		# ticks along the right edge are off
		    labelleft=False)# labels along the left edge are off

		ax.xaxis.grid(True)
		ax.set_xlim([-1,101])
		ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

		str_title = str_title
		plt.title(title if title else str_title)
		plt.show()


class CarriesProcess(Descent):
	""" Carries process formed by the cumulative sum of i.i.d. digits in :math:`\{0, \dots, b-1\}`. This is a DPP on the natural integers with a non symmetric kernel.

	:param base: 
		Base/radix

	:type base:
		int, default 10

	.. seealso::

		- :cite:`BoDiFu10`
		- :ref:`carries_process`
	"""

	def __init__(self, base=10):

		self.base = base 
		self.bernoulli_param = 0.5*(1-1/self.base)
		self.list_of_samples = []
		self.size = 100

		# self.kernel = None
		# self.kernel_eig_vecs = None

	def __str__(self):
		
		str_info = ['Carries process in base {}',
								'Number of samples = {}.']

		return '\n'.join(str_info).format(self.base,
																			len(self.list_of_samples))


	def sample(self, size=100):
		""" Compute the cumulative sum (in base :math:`b`) of a sequence of i.i.d. digits and record the position of carries.

		:param size:
			size of the sequence of i.i.d. digits in :math:`\{0, \dots, b-1\}`

		:type size:
			int
		"""

		self.size = size
		A = np.random.randint(0, self.base-1, self.size)
		B = np.mod(np.cumsum(A), self.base)

		X = np.zeros(size, dtype=bool)
		X[1:] = B[1:] < B[:-1]

		carries = np.arange(0, self.size)[X]

		self.list_of_samples.append(carries)

	def plot(self, title=''):
		"""Display the process on the real line

		:param title:
			Plot title

		:type title:
			string

		.. seealso::

			- :func:`sample <sample>`
			- :func:`plot_vs_bernoullis <plot_vs_bernoullis>`
		"""

		carries = self.list_of_samples[-1]
		len_car = len(carries)

		# Display Carries and Bernoullis
		fig, ax = plt.subplots(figsize=(19,2))

		ax.scatter(carries, np.zeros(len_car), color='blue', s=20, label='Carries')

		# Spine options
		ax.spines['bottom'].set_position('center')
		ax.spines['left'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# Ticks options
		minor_ticks = np.arange(0, self.size+1)                                            
		major_ticks = np.arange(0, self.size+1, 20)                                               
		ax.set_xticks(major_ticks)                                                       
		ax.set_xticks(minor_ticks, minor=True)
		ax.set_xticklabels(major_ticks, fontsize=15)
		ax.xaxis.set_ticks_position('bottom')

		ax.tick_params(
		    axis='y',				# changes apply to the y-axis
		    which='both',		# both major and minor ticks are affected
		    left=False,			# ticks along the left edge are off
		    right=False,		# ticks along the right edge are off
		    labelleft=False)# labels along the left edge are off

		ax.xaxis.grid(True)
		ax.set_xlim([-1,101])
		ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

		str_title = r'Realization of the carries process in base $b=${}'.format(self.base)
		plt.title(title if title else str_title)
		plt.show()
	def  plot_vs_bernoullis(self, title=''):
		return(Descent.plot_vs_bernoullis(self,title=title, process_name='Carries',str_title = r'Realization of the carries process in base $b=${} and independent Bernoullis with parameter {}'.format(self.base, r'$0.5(1-1/b)={}$'.format(self.bernoulli_param))))

class DescentProcess(Descent):
	""" This is a DPP on :math:'\{1,2,\dots,n-1}' with a non symmetric kernel appearing in (or as a limite) of the descent process on the symmetric group.

		.. seealso::

			- :cite:`BoDiFu10`
			- :ref:`carries_process`
	"""

	def __init__(self, size = 100):

		self.bernoulli_param = 0.5
		self.list_of_samples = []
		self.size = size
        	

	def __str__(self):
		
		str_info = ["Descent process",
								"Number of samples = {}."]

		return "\n".join(str_info).format(self.base,
																			len(self.list_of_samples))



	def __unif_permutation(self, N):

		tmp = np.arange(N)
		for i in range(N-1, 1, -1):
				j = np.random.randint(0, i+1)
				tmp[j], tmp[i] = tmp[i], tmp[j]

		return tmp
    
	def sample(self):

         sigma = self.__unif_permutation(self.size+1)
         X = np.zeros(self.size, dtype=bool)
         X = sigma[1:] < sigma[:-1]
         descent = np.arange(0, self.size)[X]
         self.list_of_samples.append(descent)

	def plot(self, title=""):
		"""Display the process on the real line

		:param title:
			Plot title

		:type title:
			string

		.. seealso::

			- :func:`sample <sample>`
			- :func:`plot_vs_bernoullis <plot_vs_bernoullis>`
		"""

		descent = self.list_of_samples[-1]
		len_car = len(descent)

		# Display Carries and Bernoullis
		fig, ax = plt.subplots(figsize=(19,2))

		ax.scatter(descent, np.zeros(len_car), color='blue', s=20, label='Descent')

		# Spine options
		ax.spines['bottom'].set_position('center')
		ax.spines['left'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# Ticks options
		minor_ticks = np.arange(0, self.size+1)                                            
		major_ticks = np.arange(0, self.size+1, 20)                                               
		ax.set_xticks(major_ticks)                                                       
		ax.set_xticks(minor_ticks, minor=True)
		ax.set_xticklabels(major_ticks, fontsize=15)
		ax.xaxis.set_ticks_position('bottom')

		ax.tick_params(
		    axis='y',				# changes apply to the y-axis
		    which='both',		# both major and minor ticks are affected
		    left=False,			# ticks along the left edge are off
		    right=False,		# ticks along the right edge are off
		    labelleft=False)# labels along the left edge are off

		ax.xaxis.grid(True)
		ax.set_xlim([-1,101])
		ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

		str_title = r"Realization of the descent process".format(self.base)
		plt.title(title if title else str_title)
		plt.show()

	def  plot_vs_bernoullis(self, title=''):
		return(Descent.plot_vs_bernoullis(self,title=title, process_name='Descent',str_title = r'Realization of the Decent process  and independent Bernoullis with parameter 0.5'))



class VirtualDescentProcess(Descent):
	""" This is a DPP on :math:'\{1,2,\dots,n-1}' with a non symmetric kernel appearing in (or as a limite) of the descent process on the symmetric group.

			"""

	def __init__(self, size = 100,x_0=0.5):

		if (x_0 > 1) or (x_0 <0):
			raise NameError("X_0 must be  in [0,1]")
		self.bernoulli_param = 0.5*(1-x_0**2)
		self.x_0=x_0
		self.list_of_samples = []
		self.size = size
        	
		# self.kernel = None
		# self.kernel_eig_vecs = None

	def __str__(self):
		
		str_info = ["Limitting Descent process for vitural permutations ",
								"Number of samples = {}."]

		return "\n".join(str_info).format(self.base,
																			len(self.list_of_samples))


	def __unif_permutation(self, N):

		tmp = np.arange(N)
		for i in range(N-1, 1, -1):
				j = np.random.randint(0, i+1)
				tmp[j], tmp[i] = tmp[i], tmp[j]

		return tmp
    
	def sample(self):

		sigma = self.__unif_permutation(self.size+1)
		X = np.zeros(self.size, dtype=bool)
		X = sigma[1:] < sigma[:-1]
		Y = np.random.binomial(2,self.x_0,self.size+1) != np.ones(self.size+1)
		Z = [i for i in range(self.size) if  (((not Y[i]) and Y[i+1]) or ((not Y[i]) and (not Y[i+1]) and X[i]))] 
		descent = np.arange(0, self.size)[Z]
		self.list_of_samples.append(descent)

	def plot(self, title=""):
		"""Display the process on the real line

		:param title:
			Plot title

		:type title:
			string

		.. seealso::

			- :func:`sample <sample>`
			- :func:`plot_vs_bernoullis <plot_vs_bernoullis>`
		"""

		descent = self.list_of_samples[-1]
		len_car = len(descent)

		# Display Carries and Bernoullis
		fig, ax = plt.subplots(figsize=(19,2))

		ax.scatter(descent, np.zeros(len_car), color='blue', s=20, label='Lim Descent')

		# Spine options
		ax.spines['bottom'].set_position('center')
		ax.spines['left'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# Ticks options
		minor_ticks = np.arange(0, self.size+1)                                            
		major_ticks = np.arange(0, self.size+1, 20)                                               
		ax.set_xticks(major_ticks)                                                       
		ax.set_xticks(minor_ticks, minor=True)
		ax.set_xticklabels(major_ticks, fontsize=15)
		ax.xaxis.set_ticks_position('bottom')

		ax.tick_params(
		    axis='y',				# changes apply to the y-axis
		    which='both',		# both major and minor ticks are affected
		    left=False,			# ticks along the left edge are off
		    right=False,		# ticks along the right edge are off
		    labelleft=False)# labels along the left edge are off

		ax.xaxis.grid(True)
		ax.set_xlim([-1,101])
		ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

		str_title = r"Realization of the limitting descent process".format(self.base)
		plt.title(title if title else str_title)
		plt.show()
	
	def  plot_vs_bernoullis(self, title=''):
		return(Descent.plot_vs_bernoullis(self,title=title, process_name='Descent',str_title = "Realization of the limiting descent process with parameter " +str(self.x_0)+ " and independent Bernoullis with parameter " + str(self.bernoulli_param)))
################
# Permutations #
################

class PoissonizedPlancherel:
	""" Poissonized Plancherel measure

	:param theta: 
		Base/radix

	:type theta:
		int, default 10

	.. seealso::

		- :cite:`Bor09` Section 6
	 	- :ref:`poissonized_plancherel_measure`
	"""

	def __init__(self, theta=10):

		self.theta = theta # Poisson param setting the length of the permutation
		self.list_of_samples = []

	def __str__(self):

		str_info = ('Poissonized Plancherel measure with parameter {}',
								'Number of samples = {}.')

		return '\n'.join(str_info).format(self.theta, len(self.list_of_samples))

	def info(self):
		""" Print infos about the :class:`UST` object
		"""
		print(self.__str__())

	def sample(self):
		""" Sample from the Poissonized Plancherel measure and build the associated process.
		"""

		N = np.random.poisson(self.theta)
		sigma = self.__unif_permutation(N)
		P, _ = self.__RSK(sigma)
		sampl = [len(row)-i+0.5 for i, row in enumerate(P)]
		self.list_of_samples.append(sampl)

	def __unif_permutation(self, N):

		tmp = np.arange(N)
		for i in range(N-1, 1, -1):
				j = np.random.randint(0, i+1)
				tmp[j], tmp[i] = tmp[i], tmp[j]

		return tmp

	def __RSK(self, sigma, len_1st_row=True):
		"""Perform Robinson-Schensted-Knuth correspondence on a sequence of reals, e.g. a permutation
		"""

		P, Q = [], [] # Insertion/Recording tableaux

		# Enumerate the sequence
		for it, x in enumerate(sigma):

			# Iterate along the rows of the tableau P to find a place for the bouncing x and record the position where it is inserted
			for row_P, row_Q in zip(P,Q):

				# In case x finds a place at the end of a row of P add it and record its position to the row of Q
				if x >= row_P[-1]:
					row_P.append(x); row_Q.append(it+1)
					break

				# Otherwise find the place where x must be added to keep the row ordered
				ind_insert = bisect_right(row_P, x)
				# Swap x with
				x, row_P[ind_insert] = row_P[ind_insert], x

			# In case the bouncing x cannot find a place at the end of a row of P create a new row and save
			else:
				P.append([x]); Q.append([it+1])

		return P, Q

	def plot(self, title=''):
		"""Display the process on the real line

		:param title:
			Plot title

		:type title:
			string

		.. seealso::

			- :func:`sample <sample>`
		"""

		ppDPP = self.list_of_samples[-1]
		len_pp = len(ppDPP)

		# Display the reparametrized Plancherel sample
		fig, ax = plt.subplots(figsize=(19,2))

		ax.scatter(ppDPP, np.zeros(len_pp), color='blue', s=20)

		# Spine options
		ax.spines['bottom'].set_position('center')
		ax.spines['left'].set_visible(False)
		ax.spines['top'].set_visible(False)
		ax.spines['right'].set_visible(False)

		# Ticks options

		end_ax = np.max(np.abs(ppDPP))+0.5
		minor_ticks = np.arange(-end_ax, end_ax+1)                                            
		major_ticks = np.arange(-100, 100+1, 10)                                               
		ax.set_xticks(major_ticks)                                                       
		ax.set_xticks(minor_ticks, minor=True)
		ax.set_xticklabels(major_ticks, fontsize=15)
		ax.xaxis.set_ticks_position('bottom')

		ax.tick_params(
		    axis='y',				# changes apply to the y-axis
		    which='both',		# both major and minor ticks are affected
		    left=False,			# ticks along the left edge are off
		    right=False,		# ticks along the right edge are off
		    labelleft=False)# labels along the left edge are off

		ax.xaxis.grid(True)
		ax.set_xlim([-end_ax-2, end_ax+2])
		# ax.legend(bbox_to_anchor=(0,0.85), frameon=False, prop={'size':20})

		str_title = r'Realization of the DPP associated to the poissonized Plancherel measure with parameter $\theta=${}'.format(self.theta)
		plt.title(title if title else str_title)
		plt.show()







# def r_hahn(N, a=0, b=0):

# 	if (N<=0) | (a<-1) | (b<-1):
# 		raise ValueError('Arguments(s) out of range: N>0, a,b>-1')

# 	alpha_beta_coef = np.zeros((N+1,2))
# 	ind_0_Np1 = np.arange(1,N+2)
# 	alpha_beta_coef[0,1] = np.prod(1+(a+b+1)/ind_0_Np1)

# 	if (a+b)==0:
# 		aux = ind_0_Np1
# 		alpha_beta_coef[:,0] = ((2*aux+a+b-1)*N+(b-a)*aux+a)\
# 												/(2*(2*aux-1))
# 		aux = ind_0_Np1[:-1]
# 		alpha_beta_coef[1:,1] = .25\
# 													*((N+1)**2)*(1+a/aux)*(1+b/aux)*(1-(aux/(N+1))**2)\
# 									 				/(4-(1/aux)**2);
# 	elif (a+b+1)==0:
# 		aux = ind_0_Np1
# 		alpha_beta_coef[:,0] = ((2*(aux-1)**2+b)*N+(2*b+1)*(aux-1)**2)\
# 												/(4*(aux-1)**2-1)

# 		aux = ind_0_Np1[:-1]
# 		alpha_beta_coef[1:,1] = .25\
# 													*((N+1)**2)\
# 													*(1+a/aux)\
# 													*(1+b/aux)\
# 													*(1-aux/(N+1))\
# 													*(1+(aux-1)/(N+1))\
# 													/(4-(1/aux)**2)
# 	else:
# 		aux = ind_0_Np1
# 		alpha_beta_coef[:,0]=((aux+a+b)*(aux+a)*(N-aux+1)/(2*aux+a+b)\
# 							 						+(aux-1)*(aux+b-1)*(N+aux+a+b)/(2*aux+a+b-2))\
# 												/(2*aux+a+b-1)

# 		aux = ind_0_Np1[:-1]
# 		alpha_beta_coef[1:,1]=((N+1)**2)\
# 												*(1+a/aux)*(1+b/aux)\
# 												*(1+(a+b)/aux)\
# 												*(1-aux/(N+1))\
# 												*(1+(aux+a+b)/(N+1))\
# 						 						/(((2+(a+b)/aux)**2)*((2+(a+b)/aux)**2-(1/aux)**2))

# 	return alpha_beta_coef

# #### Kravchuk
# M, p = 10, 0.5
# mu = [binom(M,p).pmf(k) for k in range(M+1)]

# def alpha_coef(M, N, p):

# 	return (1-2*p)*np.arange(N) + p*M

# def beta_coef(M, N, p):
# 	# beta_0=1.0 b_k = p*(1-p)*k*(N-k+1)
# 	tmp = np.arange(N)

# 	coef = p*(1-p)*tmp*(M-tmp+1)
# 	coef[0]=1.0

# 	return coef
