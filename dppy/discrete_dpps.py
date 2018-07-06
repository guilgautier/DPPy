# coding: utf-8
try: # Local import
	from .exact_sampling import *
	from .mcmc_sampling import *
except SystemError:
	from exact_sampling import *
	from mcmc_sampling import *
	
import matplotlib.pyplot as plt
from warnings import warn

class Discrete_DPP:

###################
### Constructor ###
###################

	def __init__(self, kernel_type, projection=False, **params):

		self.kernel_type = kernel_type 
		self.__check_kernel_type_arg()

		self.projection = projection
		self.__check_projection_arg()

		#### Parameters of the DPP
		self.params_keys = params.keys()

		### Inclusion kernel K: P(S C X) = det(K_S)
		self.K = params.get("K", None)
		# If eigendecomposition available: K_eig_dec = [eig_vals, eig_vecs]
		self.K_eig_vals, self.eig_vecs = params.get("K_eig_dec", [None, None])
		# If full row rank feature matrix passed via "A_zono" it means that there is the underlying projection kernel is K = A.T (AA.T)^-1 A. A priori, you want to use zonotope approximate sampler.
		if "A_zono" in self.params_keys:
			self.A_zono = params.get("A_zono")

		### Marginal kernel L: P(X=S) propto det(L_S) = det(L_S)/det(I+L)
		self.L = params.get("L", None)
		# If eigendecomposition available: L_eig_dec = [eig_vals, eig_vecs]
		self.L_eig_vals, self.eig_vecs = params.get("L_eig_dec", 
											[None, None if self.eig_vecs is None else self.eig_vecs])
		# If L defined as Gram matrix: L = Phi.T Phi, with feature matrix Phi dxN
		if "L_gram_factor" in self.params_keys:
			self.L_gram_factor = params.get("L_gram_factor", None) 
			# In case d<N, use "dual" view 
			self.L_dual = None # L' = Phi Phi.T
			self.L_dual_eig_vals, self.L_dual_eig_vecs = None, None

		self.__check_params_validity()

		#### Sampling
		self.sampling_mode = None
		### Exact:
		## if K (inclusion) kernel is projection
		# - ``'GS'`` for Gram-Schmidt
		## else
		# - ``'GS'``
		# - ``'GS_bis'`` slight modif of Gram-Schmidt
		# - ``'KuTa12'`` for Kulesza (Algo 1).
		### Approximate:
		## Local chains
		# - 'AED' Add-Exchange-Delete
		# - 'AD' Add-Delete
		# - 'E' Exchange
		## Zonotope
		# No argument to be passed, implicit if A_zono given
		self.list_of_samples = []

	def __str__(self):
		str_info = ("DPP defined through {}{} kernel".format(\
											"projection " if self.projection else "",
											self.kernel_type),
								"Parametrized by {}".format(self.params_keys),
								"- sampling mode = {}".format(self.sampling_mode),
								"- number of samples = {}".format(len(self.list_of_samples)))

		return "\n".join(str_info)

#############################
### Hidden object methods ###
#############################

#### Check routines

	def __check_kernel_type_arg(self):
		### Ensemble type
		if self.kernel_type not in ("inclusion", "marginal"):
			err_print = ("Invalid 'kernel_type' argument, choose among:",
									"- 'inclusion': inclusion kernel, P(S C X) = det(K_S)",
									"- 'marginal': marginal kernel, P(X=S) propto det(L_S)")
			raise ValueError("\n".join(err_print))

	def __check_projection_arg(self):
		if not isinstance(self.projection, bool):
			err_print = "Invalid 'projection' argument: must be True/False"
			raise ValueError(err_print)

	def __check_params_validity(self):

		### Check initialization parameters of the DPP

		## For inclusion kernel
		if self.kernel_type == "":

			auth_params = ("K", "K_eig_dec", "A_zono")
			if any([key in auth_params for key in self.params_keys]):

				if self.K is not None:
					self.__check_symmetry_of_kernel(self.K)

					if self.projection:
						self.__check_is_projection_kernel(self.K)

				elif self.K_eig_vals is not None:

					if self.projection:
						self.__check_eig_vals_equal_O1(self.K_eig_vals)
					else:
						self.__check_eig_vals_in_01(self.K_eig_vals)

				elif self.A_zono is not None:
					# A_zono (dxN) must be full row rank, first sanity check is d<=N
					self.__full_row_rank(self.A_zono)

			else:
				err_print = ("Invalid parameter(s) for inclusion kernel, choose among:",
										"- 'K': 0 <= K <= I", 
										"- 'K_eig_dec': (eig_vals, eig_vecs)", 
										"- 'A_zono': A is dxN matrix, with rank(A)=d corresponding to K = A.T (AA.T)^-1 A",
										"Given: {}".format(self.params_keys))
				raise ValueError("\n".join(err_print))

		## For marginal kernel
		elif self.kernel_type == "marginal":

			auth_params = ("L", "L_eig_dec", "L_gram_factor")
			if any([key in auth_params for key in self.params_keys]):

				if self.L is not None:
					self.__check_symmetry_of_kernel(self.L)

					if self.projection:
						self.__check_is_projection_kernel(self.L)

				elif self.L_eig_vals is not None:
						self.__check_eig_vals_geq_0(self.L_eig_vals)
						# self.__check_eig_vals_equal_O1(self.L_eig_vals)

				elif self.L_gram_factor is not None: # 
					self.__check_L_dual_or_not(self.L_gram_factor)

					if self.projection:
						warn("'L_gram_factor'+'projection'=True is a very weird setting, you may switch to 'projection'=False")

			else:
				err_print = ("Invalid parameter(s) for marginal kernel, choose among:",
										"- 'L': L >= 0", 
										"- 'L_eig_dec': (eig_vals, eig_vecs)", 
										"- 'L_gram_factor': Phi is dxN feature matrix corresponding to L = Phi.T Phi",
										"Given: {}".format(self.params_keys))
				raise ValueError("\n".join(err_print))

	def __check_symmetry_of_kernel(self, kernel):
		if not np.allclose(kernel.T, kernel):
			err_print = "Invalid kernel: not symmetric"
			raise ValueError(err_print)

	def __check_is_projection_kernel(self, kernel):
			# Cheap test checking reproducing property
			nb_tmp = 5
			items_to_check = np.arange(nb_tmp)
			K_i_ = kernel[items_to_check, :]
			K_ii = kernel[items_to_check, items_to_check]

			if np.allclose(np_inner1d(K_i_, K_i_), K_ii):
				pass
			else:
				raise ValueError("Invalid kernel: doesn't seem to be a projection")

	def __check_eig_vals_equal_O1(self, eig_vals):

		tol = 1e-8
		eig_vals_close_to_0 = (-tol<=eig_vals) & (eig_vals<=tol)
		eig_vals_close_to_1 = (1-tol<=eig_vals) & (eig_vals<=1+tol)

		if np.all(eig_vals_close_to_0 ^ eig_vals_close_to_1):
			pass
		else:
			raise ValueError("Invalid kernel: doesn't seem to be a projection, check that the eigenvalues provided are equal to 0 or 1")

	def __check_eig_vals_in_01(self, eig_vals):

		tol = 1e-8

		if not np.all((-tol<=eig_vals) & (eig_vals<=1.0+tol)):
			err_print = "Invalid kernel for inclusion kernel, eigenvalues not in [0,1]"
			raise ValueError(err_print)
			
	def __check_eig_vals_geq_0(self, eig_vals):

		tol = 1e-8

		if not np.all(eig_vals>=-tol):
			err_print = "Invalid kernel for marginal kernel, eigenvalues not >= 0"
			raise ValueError(err_print)

	def __full_row_rank(self, A_zono):

		d, N = A_zono.shape
		rank = np.linalg.matrix_rank(A_zono)

		if rank == d:
			if not self.projection: 
				warn("Weird setting: inclusion kernel defined via 'A_zono' but 'projection'=False. 'projection' switched to True")
				self.projection = True

		else:
			err_print = ("Invalid 'A_zono' (dxN) parameter, not full row rank: d(={}) != rank(={})".format(d, rank))
			raise ValueError(err_print)

	def __check_L_dual_or_not(self, L_gram_factor):

		d, N = L_gram_factor.shape
		
		if d<N:
			self.L_dual = L_gram_factor.dot(L_gram_factor.T)
			str_print = "d={} < N={}: L dual kernel was computed".format(d, N)

		else:
			self.L = L_gram_factor.T.dot(L_gram_factor)
			str_print = "d={} >= N={}: L kernel was computed".format(d, N)

		print(str_print)

### Eigendecomposition

	def __eigendecompose(self, kernel):
		# print("Eigendecomposition was performed")
		return la.eigh(kernel)

######################
### Object methods ###
######################

	def info(self):
		print(self.__str__())

	def flush_samples(self):
		self.list_of_samples = []

	### Exact sampling
	def sample_exact(self, sampling_mode="GS"):
		""" Sample exactly from the corresponding :class:`Discrete_DPP <Discrete_DPP>` object

		:param sampling_mode:

			If ``projection=True``:
				- ``'GS'`` (default): Gram-Schmidt on the columns of :math:`\mathbf{K}`
			
			If ``projection=False``:
				- ``'GS'`` (default): 
				- ``'GS_bis'``: Slight modification of ``'GS'``
				- ``'KuTa12'``: Algorithm 1 in :cite:`KuTa12`
		:type sampling_mode:
			string, default ``'GS'``

		.. seealso::

			- :ref:`discrete_dpps_exact_sampling`
			- :func:`sample_mcmc <sample_mcmc>`
		"""

		self.sampling_mode = sampling_mode

		## If eigen decoposition of K, L or L_dual is available USE IT!
		if self.K_eig_vals is not None:
			self.__check_eig_vals_in_01(self.K_eig_vals)
			# Phase 1
			V = dpp_eig_vecs_selector(self.K_eig_vals, self.eig_vecs)
			# Phase 2
			sampl = dpp_sampler_eig(V, self.sampling_mode)
			self.list_of_samples.append(sampl)

		elif self.L_eig_vals is not None:
			self.K_eig_vals = self.L_eig_vals/(1.0+self.L_eig_vals)
			self.sample_exact(self.sampling_mode)

		elif "L_gram_factor" in self.params_keys: 
		# If DPP is marginal kernel with parameter "L_gram_factor" i.e. L = Phi.T Phi but dual kernel L' = Phi Phi.T was cheaper to use (computation of L' and diagonalization for sampling)
			if self.L_dual_eig_vals is not None:
				# Phase 1
				V = dpp_eig_vecs_selector_L_dual(self.L_dual_eig_vals, 
																				self.L_dual_eig_vecs,
																				self.L_gram_factor)
				# Phase 2
				sampl = dpp_sampler_eig(V, self.sampling_mode)
				self.list_of_samples.append(sampl)

			elif self.L_dual is not None:
				self.L_dual_eig_vals, self.L_dual_eig_vecs\
										= self.__eigendecompose(self.L_dual)
				self.__check_eig_vals_geq_0(self.L_dual_eig_vals)
				self.sample_exact(self.sampling_mode)

		## Otherwise
		# If DPP is inclusion kernel with projection kernel no need of eigendecomposition, you can apply Gram-Schmidt on the columns of K (equiv rows because of symmetry)
		elif (self.K is not None) and self.projection:
			sampl = proj_dpp_sampler_kernel(self.K, self.sampling_mode)
			self.list_of_samples.append(sampl)

		# If DPP is inclusion kernel with generic kernel, eigen-decompose it
		elif self.K is not None:
			self.K_eig_vals, self.eig_vecs = self.__eigendecompose(self.K)
			self.sample_exact(self.sampling_mode)

		# If DPP is marginal kernel with kernel L, eigen-decompose it
		elif self.L is not None:
			self.L_eig_vals, self.eig_vecs = self.__eigendecompose(self.L)
			self.sample_exact(self.sampling_mode)

		# If DPP is inclusion kernel with parameter 'A_zono', a priori you wish to use the zonotope approximate sampler: warning is raised 
		# But corresponding projection kernel K = A.T (AA.T)^-1 A is computed 
		elif "A_zono" in self.params_keys:
			warn("DPP defined via 'A_zono', apriori you want to use 'sampl_mcmc', but you have called 'sample_exact'")
			self.compute_K()
			self.projection, self.sampling_mode = True, "GS"
			self.sample_exact(self.sampling_mode)

	### Approximate sampling
	def sample_mcmc(self, sampling_mode, **params):

		auth_sampl_mod = ("AED", "AD", "E", "zonotope")
		# AED: 

		if sampling_mode in auth_sampl_mod:
			self.sampling_mode = sampling_mode

			if self.sampling_mode == "zonotope":

				if "A_zono" in self.params_keys:
					MC_samples = zonotope_sampler(self.A_zono, **params)

				else:
					err_print = ("Invalid 'sampling_mode':",
											"DPP must be defined via 'A_zono' to use 'zonotope' as sampling mode")
					raise ValueError("\n".join(err_print))

			elif self.sampling_mode == "E":

				if (self.kernel_type == "inclusion") and self.projection:
					self.compute_K()
					# |sample|=Tr(K) a.s. for projection DPP(K)
					params.update({'size': int(np.round(np.trace(self.K)))}) 

					MC_samples = dpp_sampler_mcmc(self.K, self.sampling_mode, **params)

				else:
					self.compute_L()
					MC_samples = dpp_sampler_mcmc(self.L, self.sampling_mode, **params)

			elif self.sampling_mode in ("AED", "AD"):
				self.compute_L()
				MC_samples = dpp_sampler_mcmc(self.L, self.sampling_mode, **params)

			self.list_of_samples.append(MC_samples)

		else:
			err_print = ("Invalid 'sampling_mode' parameter, choose among:",
									"- 'AED' for Add-Exchange-Delete",
									"- 'AD' for Add-Delete",
									"- 'E' for Exchange",
									"- 'zonotope' for zonotope sampler (projection inclusion kernel only)",
									"Given 'sampling_mode' = {}".format(sampling_mode))
			raise ValueError("\n".join(err_print))

	def compute_K(self, msg=None):
		"""K = L(I+L)^-1 = I - (I+L)^-1"""
		if self.K is None:
			if not msg:
				print("K (inclusion) kernel computed via:")

			if "A_zono" in self.params_keys:
				print("- 'A_zono' i.e. K = A.T (AA.T)^-1 A")
				A = self.A_zono
				self.K = A.T.dot(np.linalg.inv(A.dot(A.T))).dot(A)

			elif self.K_eig_vals is not None:
				print("- U diag(eig_K) U.T")
				self.K = (self.eig_vecs * self.K_eig_vals).dot(self.eig_vecs.T)

			elif self.L_eig_vals is not None:
				print("- eig_K = eig_L/(1+eig_L)")
				self.K_eig_vals = self.L_eig_vals/(1.0 + self.L_eig_vals)
				self.compute_K(msg=True)

			elif self.L is not None:
				print("- eigendecomposition of L")
				self.L_eig_vals, self.eig_vecs = self.__eigendecompose(self.L)
				self.__check_eig_vals_geq_0(self.L_eig_vals)
				self.compute_K(msg=True)

			else:
				self.compute_L(msg=True)
				self.compute_K(msg=True)

		else:
			print("K (inclusion) kernel available")

	def compute_L(self, msg=False):
		"""L = K(I-K)^-1 = (I-K)^-1 - I"""
		if (self.kernel_type == "inclusion") and self.projection:
			err_print = ("L = K(I-K)^-1 = (I-K)^-1 - I kernel cannot be computed:",
									"K being a projection kernel it has some eigenvalues equal to 1")
			raise ValueError("\n".join(err_print))

		elif self.L is None:
			if not msg:
				print("L (marginal) kernel computed via:")

			if "L_gram_factor" in self.params_keys:
				print("- 'L_gram_factor' i.e. L = Phi.T Phi")
				self.L = self.L_gram_factor.T.dot(self.L_gram_factor)

			elif self.L_eig_vals is not None:
				print("- U diag(eig_L) U.T")
				self.L = (self.eig_vecs * self.L_eig_vals).dot(self.eig_vecs.T)

			elif self.K_eig_vals is not None:
				try: # to compute eigenvalues of kernel L = K(I-K)^-1
					print("- eig_L = eig_K/(1-eig_K)")
					np.seterr(divide='raise')
					self.L_eig_vals = self.K_eig_vals/(1.0 - self.K_eig_vals)
					self.compute_L(msg=True)
				except:
					err_print = ("Eigenvalues of L kernel cannot be computed",
											"eig_L = eig_K/(1-eig_K)",
											"K kernel has some eig_K very close to 1.",
											"Hint: 'K' kernel might be a projection.")
					raise FloatingPointError("\n".join(err_print))

			elif self.K is not None:
				print("- eigendecomposition of K")
				self.K_eig_vals, self.eig_vecs = self.__eigendecompose(self.K)
				self.__check_eig_vals_in_01(self.K_eig_vals)
				self.compute_L(msg=True)

			else:
				self.compute_K(msg=True)
				self.compute_L(msg=True)

		else:
			print("L (marginal) kernel available")



	def plot(self):
		"""Display a heatmap of the kernel"""

		fig, ax = plt.subplots(1,1)

		if self.kernel_type == "inclusion":
			if self.K is None:
				self.compute_K()
			self.nb_items = self.K.shape[0]
			kernel_to_plot = self.K
			str_print = "K (inclusion) kernel"

		elif self.kernel_type == "marginal":
			if self.L is None:
				self.compute_L()
			self.nb_items = self.L.shape[0]
			kernel_to_plot = self.L
			str_print = "L (marginal) kernel"

		print(str_print)
		heatmap = ax.pcolor(kernel_to_plot, cmap="jet")

		ax.set_aspect("equal")

		ticks = np.arange(self.nb_items)
		ticks_label = [r"${}$".format(tic) for tic in ticks]

		ax.xaxis.tick_top()
		ax.set_xticks(ticks+0.5, minor=False)

		ax.invert_yaxis()
		ax.set_yticks(ticks+0.5, minor=False)

		ax.set_xticklabels(ticks_label, minor=False)
		ax.set_yticklabels(ticks_label, minor=False)

		plt.colorbar(heatmap)
		plt.show()