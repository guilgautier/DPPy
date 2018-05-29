# coding: utf-8
from .exact_sampling import *
from .approximate_sampling import *
import matplotlib.pyplot as plt
from warnings import warn

class Discrete_DPP:

###################
### Constructor ###
###################

	def __init__(self, ensemble_type, projection=False, **params):

		self.ensemble_type = ensemble_type 
		self.__check_ensemble_type_arg()

		self.projection = projection
		self.__check_projection_arg()

		#### Parameters of the DPP
		self.params_keys = params.keys()

		### Inclusion kernel K: P(S C X) = det(K_S)
		self.K = params.get("K_kernel", None)
		# If eigendecomposition available: K_eig_dec = [eig_vals, eig_vecs]
		self.K_eig_vals, self.eig_vecs = params.get("K_eig_dec", [None, None])
		# If full row rank feature matrix passed via "A_zono" it means that there is the underlying projection kernel is K = A.T (AA.T)^-1 A. A priori, you want to use zonotope approximate sampler.
		if "A_zono" in self.params_keys:
			self.A_zono = params.get("A_zono")

		### Marginal kernel L: P(X=S) propto det(L_S) = det(L_S)/det(I+L)
		self.L = params.get("L_kernel", None)
		# If eigendecomposition available: L_eig_dec = [eig_vals, eig_vecs]
		self.L_eig_vals, self.eig_vecs = params.get("L_eig_dec", [None, None])
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
		# - 'GS' for Gram-Schmidt
		## else
		# - 'GS'
		# - 'GS_bis' slight modif of Gram-Schmidt
		# - 'KuTa12' for Kulesza (Algo 1).
		### Approximate:
		## Local chains
		# - 'AED' Add-Exchange-Delete
		# - 'AD' Add-Delete
		# - 'E' Exchange
		## Zonotope
		# No argument to be passed, implicit if A_zono given
		self.list_of_samples = []

	def __str__(self):
		str_info = ("DPP defined as {}-ensemble {}".format(\
											self.ensemble_type, 
											"with projection kernel" if self.projection else ""),
								"Parametrized by {}".format(self.params_keys),
								"- sampling mode = {}".format(self.sampling_mode),
								"- number of samples = {}".format(len(self.list_of_samples)))

		return "\n".join(str_info)

#############################
### Hidden object methods ###
#############################

#### Check routines

	def __check_ensemble_type_arg(self):
		### Ensemble type
		if self.ensemble_type not in ("K", "L"):
			err_print = ("Invalid 'ensemble_type' argument, choose among:",
									"- 'K': inclusion kernel, P(S C X) = det(K_S)",
									"- 'L': marginal kernel, P(X=S) propto det(L_S)")
			raise ValueError("\n".join(err_print))

	def __check_projection_arg(self):
		if not isinstance(self.projection, bool):
			err_print = "Invalid 'projection' argument: must be True/False"
			raise ValueError(err_print)

	def __check_params_validity(self):

		### Check initialization parameters of the DPP

		## For K-ensemble
		if self.ensemble_type == "K":

			auth_params = ("K_kernel", "K_eig_dec", "A_zono")
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
				err_print = ("Invalid parameter(s) for K-ensemble, choose among:",
										"- 'K_kernel': 0 <= K <= I", 
										"- 'K_eig_dec': (eig_vals, eig_vecs)", 
										"- 'A_zono': A is dxN matrix, with rank(A)=d corresponding to K = A.T (AA.T)^-1 A",
										"Given: {}".format(self.params_keys))
				raise ValueError("\n".join(err_print))

		## For L-ensemble
		elif self.ensemble_type == "L":

			auth_params = ("L_kernel", "L_eig_dec", "L_gram_factor")
			if any([key in auth_params for key in self.params_keys]):

				if self.L is not None:
					self.__check_symmetry_of_kernel(self.L)

					if self.projection:
						self.__check_is_projection_kernel(self.L)

				elif self.L_eig_vals is not None:

					if self.projection:
						self.__check_eig_vals_equal_O1(self.L_eig_vals)
					else:
						self.__check_eig_vals_geq_0(self.L_eig_vals)

				elif self.L_gram_factor is not None: # 
					self.__check_L_dual_or_not(self.L_gram_factor)

			else:
				err_print = ("Invalid parameter(s) for L-ensemble, choose among:",
										"- 'L_kernel': L >= 0", 
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

			if not np.allclose(np_inner1d(K_i_, K_i_), K_ii):
				raise ValueError("Invalid kernel: doesn't seem to be a projection")

	def __check_eig_vals_equal_O1(self, eig_vals):

		tol = 1e-8
		eig_vals_close_to_0 = (-tol<=eig_vals) & (eig_vals<=tol)
		eig_vals_close_to_1 = (1-tol<=eig_vals) & (eig_vals<=1+tol)

		if not np.all(eig_vals_close_to_0 ^ eig_vals_close_to_1):
			ValueError("Invalid kernel: doesn't seem to be a projection")

	def __check_eig_vals_in_01(self, eig_vals):

		tol = 1e-8

		if not np.all((-tol<=eig_vals) & (eig_vals<=1.0+tol)):
			err_print = "Invalid kernel for K-ensemble, eigenvalues not in [0,1]"
			raise ValueError(err_print)
			
	def __check_eig_vals_geq_0(self, eig_vals):

		tol = 1e-8

		if not np.all(eig_vals>=-tol):
			err_print = "Invalid kernel for L-ensemble, eigenvalues not >= 0"
			raise ValueError(err_print)

	def __full_row_rank(self, A_zono):

		d, N = A_zono.shape
		rank = np.linalg.matrix_rank(A_zono)

		if rank == d:
			if not self.projection: 
				warn("'projection' attribute set to True")
				self.projection = True

		else:
			err_print = ("Invalid 'A_zono' (dxN) parameter, not full row rank: d(={}) != rank(={})".format(d, rank))
			raise ValueError(err_print)



	def __check_L_dual_or_not(self):

		d, N = self.L_gram_factor
		
		if d<N:
			self.L_dual = self.L_gram_factor.dot(self.L_gram_factor.T)
			str_print = "d={} < N={}: L dual kernel was computed".format(d, N)

		else:
			self.L = L_gram_factor.T.dot(self.L_gram_factor)
			str_print = "d={} >= N={}: L kernel was computed".format(d, N)

### Eigendecomposition

	def __eigendecompose(self, kernel):
		print("Eigendecomposition was performed")
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

		self.sampling_mode = sampling_mode

		# If eigen decoposition of K, L or L_dual is available USE IT!
		if self.K_eig_vals is not None:
			# Phase 1
			V = dpp_eig_vecs_selector(self.K_eig_vals, self.eig_vecs)
			# Phase 2
			sampl = dpp_sampler_eig(V, self.sampling_mode)

		elif self.L_eig_vals is not None:
			self.K_eig_vals = self.L_eig_vals/(1.0+self.L_eig_vals)
			self.sample_exact(self.sampling_mode)

		elif "L_gram_factor" in self.params_keys:
			if self.L_dual_eig_vals is not None:
				# Phase 1
				V = dpp_eig_vecs_selector_L_dual(self.L_dual_eig_vals, 
																				self.L_dual_eig_vecs,
																				self.L_gram_factor)
				# Phase 2
				sampl = dpp_sampler_eig(V, self.sampling_mode)

			elif self.L_dual is not None:
				self.L_dual_eig_vals, self.L_dual_eig_vecs\
										= self.__eigendecompose(self.L_dual)
				self.sample_exact(self.sampling_mode)

		if (self.K is not None) and self.projection:
			sampl = proj_dpp_sampler_kernel(self.K)

		elif self.K is not None:
			self.K_eig_vals, self.eig_vecs = self.__eigendecompose(self.K)
			self.sample_exact(self.sampling_mode)

		elif self.L is not None:
			self.L_eig_vals, self.eig_vecs = self.__eigendecompose(self.L)
			self.sample_exact(self.sampling_mode)

		elif "A_zono" in self.params_keys:
			warn("DPP defined via 'A_zono', apriori you want to use 'sample_approx', but you have called 'sample_exact'")
			self.compute_K_kernel()
			self.sampling_mode(self.sampling_mode)

		self.list_of_samples.append(sampl)

	### Approximate sampling
	def sample_approx(self, sampling_mode, **params):

		auth_sampl_mod = ("AED", "AD", "E", "zonotope")

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

				if (self.ensemble_type == "K") and self.projection:
					self.compute_K_kernel()
					# |sample|=Tr(K) a.s. for projection DPP(K)
					params.update({'size': int(np.round(np.trace(self.K)))}) 

					MC_samples = dpp_sampler_approx(self.K, self.sampling_mode, **params)

				else:
					self.compute_L_kernel()
					MC_samples = dpp_sampler_approx(self.L, self.sampling_mode, **params)

			elif self.sampling_mode in ("AED", "AD"):
				self.compute_L_kernel()
				MC_samples = dpp_sampler_approx(self.L, self.sampling_mode, **params)

			self.list_of_samples.append(MC_samples)

		else:
			err_print = ("Invalid 'sampling_mode' parameter, choose among:",
									"- 'AED' for Add-Exchange-Delete",
									"- 'AD' for Add-Delete",
									"- 'E' for Exchange",
									"- 'zonotope' for zonotope sampler (projection K-ensemble only)",
									"Given 'sampling_mode' = {}".format(sampling_mode))
			raise ValueError("\n".join(err_print))

	def compute_K_kernel(self, msg=None):
		"""K = L(I+L)^-1 = I - (I+L)^-1"""
		if self.K is None:
			str_print = "K kernel computed via:\n" if msg is None else msg

			if "A_zono" in self.params_keys:
				A = self.A_zono
				self.K = A.T.dot(np.linalg.inv(A.dot(A.T))).dot(A)
				str_print += "- 'A_zono' i.e. K = A.T (AA.T)^-1 A"

			elif self.K_eig_vals is not None:
				str_print += "- 'K_eig_vals' i.e. U D U.T"
				self.K = (self.eig_vecs * self.K_eig_vals).dot(self.eig_vecs.T)

			elif self.L_eig_vals is not None:
				str_print += "- 'L_eig_vals' i.e eig_K = eig_L/(1+eig_L)\n"
				self.K_eig_vals = self.L_eig_vals/(1.0 + self.L_eig_vals)
				self.compute_K_kernel(msg=str_print)

			elif self.L is not None:
				self.compute_L_kernel(msg=str_print)

			print(str_print)

	def compute_L_kernel(self, msg=None):
		"""L = K(I-K)^-1 = (I-K)^-1 - I"""
		if (self.ensemble_type == "K") and self.projection:
			err_print = ("L = K(I-K)^-1 = (I-K)^-1 - I kernel cannot be computed:",
									"K being a projection kernel it has some eigenvalues equal to 1")
			raise ValueError("\n".join(err_print))

		elif self.L is None:
			str_print = "L kernel computed via:\n" if msg is None else msg

			if "L_gram_factor" in self.params_keys:
				str_print += "- 'L_gram_factor' i.e. L = Phi.T Phi"
				self.L = self.L_gram_factor.T.dot(self.L_gram_factor)

			elif self.L_eig_vals is not None:
				str_print += "- 'L_eig_vals' i.e. U D U.T\n"
				self.L = (self.eig_vecs * self.L_eig_vals).dot(self.eig_vecs.T)

			elif self.K_eig_vals is not None:
				try: # to compute eigenvalues of kernel L = K(I-K)^-1
					str_print += "- 'K_eig_vals' i.e. U D/(1-D) U.T\n"
					np.seterr(divide='raise')
					self.L_eig_vals = self.K_eig_vals/(1.0 - self.K_eig_vals)
					self.compute_L_kernel(str_print)
				except:
					err_print = ("Eigenvalues of L kernel cannot be computed",
											"eig_L = eig_K/(1-eig_K)",
											"K kernel has some eig_K very close to 1.",
											"Hint: 'K' kernel might be a projection.")
					raise FloatingPointError("\n".join(err_print))

			elif self.K is not None:
				self.K_eig_vals, self.eig_vecs = self.__eigendecompose(self.K)
				self.compute_L_kernel(msg=str_print)
			else:
				self.compute_K_kernel(msg=str_print)

			print(str_print)

	def plot(self):
		"""Display a heatmap of the kernel"""

		fig, ax = plt.subplots(1,1)

		if self.ensemble_type == "K":
			if self.K is None:
				self.compute_K_kernel()
				self.plot()
			kernel_to_plot = self.K
			str_print = "Inclusion kernel 'K'"

		elif self.ensemble_type == "L":
			if self.L is None:
				self.compute_L_kernel()
				self.plot()
			kernel_to_plot = self.L
			str_print = "Marginal kernel 'L'"

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