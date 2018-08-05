# coding: utf-8

import numpy as np
import scipy.linalg as la
import networkx as nx
import matplotlib.pyplot as plt
from itertools import chain

try: # Local import
	from .exact_sampling import proj_dpp_sampler_eig_GS
except (SystemError, ImportError):
	from exact_sampling import proj_dpp_sampler_eig_GS

##############################
### Uniform Spanning trees ###
##############################

class UST:

	def __init__(self, graph):
		
		self.graph = graph
		self.nb_nodes = len(self.graph)
		self.nb_edges = self.graph.number_of_edges()

		self.neighbors = [list(graph.neighbors(v)) for v in range(self.nb_nodes)]
		self.edges = list(self.graph.edges())
		
		self.list_of_samples = []

		self.kernel = None
		self.kernel_eig_vecs = None
		#degrees = [g.degree(v) for v in nodesices]
		#transition_proba = [[1./deg]*deg for deg in degrees]

	def sample(self, mode="Wilson"):

		if mode=="Wilson":
			sampl = self.__wilson()

		elif mode=="Aldous":
			sampl = self.__aldous()

		elif mode=="exact_finite_dpp":

			if self.kernel_eig_vecs is None:
				self.__compute_kernel_eig_vecs()

			dpp_sample = proj_dpp_sampler_eig_GS(self.kernel_eig_vecs)
			
			g_finite_dpp = nx.Graph()
			edges_finite_dpp = [self.edges[ind] for ind in dpp_sample]
			g_finite_dpp.add_edges_from(edges_finite_dpp)

			sampl = g_finite_dpp

		else:
			raise ValueError("In valid 'sampling_mode' argument. Choose among 'Wilson', 'Aldous' or 'exact_finite_dpp'.\nGiven {}".format(sampling_mode))

		self.list_of_samples.append(sampl)
		
	def flush_samples(self):
		""" Empty the ``BetaEnsemble.list_of_samples`` attribute.
		"""
		self.list_of_samples = []
		
	def compute_kernel(self):

		if self.kernel is None:
			self.__compute_kernel_eig_vecs()
			self.kernel = self.kernel_eig_vecs@self.kernel_eig_vecs.T
		else:
			print("Kernel available")
			
	def __compute_kernel_eig_vecs(self):

		if self.kernel_eig_vecs is None:
			A = nx.incidence_matrix(self.graph, oriented=True)[:-1,:].toarray()
			self.kernel_eig_vecs, _ = la.qr(A.T, mode="economic")
		else:
			pass

	def plot_kernel(self):

		if self.kernel is None: self.compute_kernel()

		fig, ax = plt.subplots(1,1)

		heatmap = ax.pcolor(self.kernel, cmap="jet")

		ax.set_aspect("equal")

		ticks = np.arange(self.nb_edges)
		ticks_label = [r"${}$".format(tic) for tic in ticks]

		ax.xaxis.tick_top()
		ax.set_xticks(ticks+0.5, minor=False)

		ax.invert_yaxis()
		ax.set_yticks(ticks+0.5, minor=False)

		ax.set_xticklabels(ticks_label, minor=False)
		ax.set_yticklabels(ticks_label, minor=False)

		# plt.title(str_title, y=1.1)

		plt.colorbar(heatmap)
		plt.show()

	def plot_sample(self):

		graph_to_plot = self.list_of_samples[-1]

		fig = plt.figure(figsize=(4,4))
		nx.draw_circular(graph_to_plot, node_color='orange', with_labels = True)

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


###################
# Carries process #
###################

def carries_process(N, b=10):

	A = np.random.randint(0, b-1, N)
	B = np.mod(np.cumsum(A), b)

	X = np.zeros(N, dtype=bool)
	X[1:] = B[1:] < B[:-1]

	carries = np.arange(0, N)[X]

	return carries

################
# Permutations #
################

def unif_permutation(N):

	N=10
	tmp = np.arange(N)
	for i in range(N-1,1,-1):
			j = np.random.randint(0, i+1)
			tmp[j], tmp[i] = tmp[i], tmp[j]

	return tmp

#sigma = [2,2,4,3,8,7,3,2]

### RSK
def RSK(sigma):
	"""Perform Robinson-Schensted-Knuth correspondence on a sequence of reals, e.g. a permutation
	"""

	P, Q = [], [] # Insertion/Recording tableaux

	# Enumerate the sequence
	for it, x in enumerate(sigma):

		# Iterate along the rows of the tableau P
		# to find a place for the bouncing x and
		# record the position where it is inserted
		for row_P, row_Q in zip(P,Q):

			# In case x finds a place at the end of a row of P
			# Add it and record its position to the row of Q
			if x >= row_P[-1]:
				row_P.append(x); row_Q.append(it+1)
				break

			# Otherwise find the place where x must be added
			# to keep the row ordered
			ind_insert = bisect_right(row_P, x)
			# Swap x with
			x, row_P[ind_insert] = row_P[ind_insert], x

		# In case the bouncing x cannot find a place at the end of a row of P
		# Create a new row and save
		else:
			P.append([x]); Q.append([it+1])

	return P, Q, len(P[0])



# def r_hahn(N, a=0, b=0):

# 	if (N<=0) | (a<-1) | (b<-1):
# 		raise ValueError("Arguments(s) out of range: N>0, a,b>-1")

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