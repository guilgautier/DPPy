# coding: utf-8
from .exact_sampling import *
from .approximate_sampling import *
import warnings
import matplotlib.pyplot as plt


class Discrete_DPP:

	def __init__(self, kernel, ensemble_type, projection_kernel=False):

		self.nb_items = kernel.shape[0]

		self.ensemble_type = ensemble_type
		self.__check_ensemble_type_validity()

		self.projection_kernel = projection_kernel
		self.__check_projection_kernel_validity(kernel)

		self.K = None 
		self.L = None
		# If valid kernel, diagonalization only for non projection kernel.
		self.eig_vals_K = None
		self.eig_vals_L = None
		self.eig_vecs = None
		self.eigendecomposition_available = False
		self.__check_kernel_for_dpp_validity(kernel) 

		self.sampling_mode = "GS" 
		### Exact sampling
		# 'GS' for Gram-Schmidt, 
		# 'Schur' for Schur complement 
		# 'KuTa12' for Kulesza (Algo 1).
		### Approx sampling
		# 'AED' 'AD' 'E' A=Add E=Exchange D=Delete
		self.list_of_samples = []

	def __str__(self):
		return self._str_info()

	def __check_ensemble_type_validity(self):

		if self.ensemble_type not in ('K', 'L'):
			str_list = ["Invalid ensemble_type parameter, use:",
									"- 'K' for inclusion probability kernel",
									"- 'L' for marginal kernel.",
									"Given: ensemble_type = '{}'".format(self.ensemble_type)]

			raise ValueError("\n".join(str_list))

	def __check_projection_kernel_validity(self, kernel):
		# Recall we first checked the kernel to be symmetric (K=K.T), here we check the reproducing property of the (orthogonal) projection kernel.
		# For this we perform a cheap test to check K^2 = K K.T = K
		if not isinstance(self.projection_kernel, bool):
			raise ValueError("Invalid projection_kernel argument: must be True/False.\nGiven projection_kernel = {}".format(self.projection_kernel))

	def __check_kernel_for_dpp_validity(self, kernel):
		"""Check symmetry, projection, and validity:
		- For K-ensemble 0<=K<=I
		- For L-ensemble L>=0"""

		# Check symmetry: kernel.T = kernel
		if not np.allclose(kernel, kernel.T):
			raise ValueError("Invalid kernel: not symmetric")

		# Check if 'K' kernel is orthogonal projection K^2 = K K.T = K
		# Only case for which eigendecomposition is not necessary
		if (self.ensemble_type == 'K') & (self.projection_kernel):
			# Cheap test checking reproducing property
			nb_tmp = 5
			items_to_check = np.arange(nb_tmp)
			K_i_ = kernel[items_to_check, :]
			K_ii = kernel[items_to_check, items_to_check]

			if np.allclose(np_inner1d(K_i_, K_i_), K_ii):
					self.K = kernel

			else:
				raise ValueError("Invalid kernel: kernel doesn't seem to be a projection")

		else:
			# Eigendecomposition necessary for non projection kernels
			eig_vals, eig_vecs = la.eigh(kernel)
			self.eigendecomposition_available = True
			tol = 1e-8 # tolerance on the eigenvalues

			# If K-ensemble
			if self.ensemble_type == 'K': 
				# Check 0 <= K <= I
				if np.all((-tol <= eig_vals) & (eig_vals <= 1.0+tol)):
					self.K = kernel
					self.eig_vals_K = eig_vals
					self.eig_vecs = eig_vecs

					try:
						np.seterr(divide='raise')
						self.eig_vals_L = self.eig_vals_K/(1.0 - self.eig_vals_K)

					except FloatingPointError as e:
						str_list = ["WARNING: {}.".format(e),
												"Eigenvalues of 'L' kernel (L=K(I-K)^-1) cannot be computed.",
												"'K' kernel has some eigenvalues equal are very close to 1.",
												"Hint: 'K' kernel might be a projection."]
						print("\n".join(str_list))
					
				else:
					raise ValueError("Invalid kernel for K-ensemble. Eigen values are not in [0,1]")

			# If L-ensemble
			elif self.ensemble_type == 'L':
				# Check L >= 0
				if np.all(eig_vals >= -tol):
					self.L = kernel
					self.eig_vals_L = eig_vals
					self.eig_vecs = eig_vecs

					self.eig_vals_K = self.eig_vals_L/(1.0 + self.eig_vals_L)
				else:
					raise ValueError("Invalid kernel for L-ensemble. Eigen values !>= 0")

	def _str_info(self, size=False):

		str_info = ["Discrete {} defined by:".format("k-DPP with k={}".format(size) 																					if size else "DPP"), 
								"- {}-ensemble on {} items",
								"- Projection kernel: {}",
								"- sampling mode = {}",
								"- number of samples = {}"]

		return "\n".join(str_info).format(self.ensemble_type, self.nb_items,
																		"Yes" if self.projection_kernel else "No",
																		self.sampling_mode,
																		len(self.list_of_samples))

	def info(self):
		print(self.__str__())

	def _compute_K_kernel(self):
		"""K = L(I+L)^-1 = I - (I+L)^-1"""
		if self.K:
			raise ValueError("'K' kernel is already available")

		elif self.L: # Diagonalization was already performed
			self.K = (self.eig_vecs * self.eig_vals_K).dot(self.eig_vecs.T)

	def _compute_L_kernel(self):
		"""L = K(I-K)^-1 = (I-K)^-1 - I"""
		if self.L:
			raise ValueError("'L' kernel is already available")

		elif self.K:
			if self.projection_kernel:
				raise ValueError("Cannot compute L=K(I-K)^-1 kernel from K since K is projection kernel => I-K not invertible")

			else:
				self.L = (self.eig_vecs * self.eig_vals_L).dot(self.eig_vecs.T)

	def flush_samples(self):
		self.list_of_samples = []

	### Exact sampling
	def sample_exact(self, sampling_mode="GS"):

		self.sampling_mode = sampling_mode

		if self.eigendecomposition_available: # If eigendecomposition available use it!
			sampl = dpp_sampler_eig(self.eig_vals_K, 
															self.eig_vecs, 
															self.sampling_mode)
		elif (self.ensemble_type == 'K') & (self.projection_kernel): 
			# If K is orthogonal projection no need for eigendecomposition, update conditional via Gram-Schmidt on columns (equiv on rows) of K
			sampl = proj_dpp_sampler_kernel(self.K, 
																			self.sampling_mode)

		else:
			raise ValueError("WARNING sampling!!")



		self.list_of_samples.append(sampl)

	### Approximate sampling
	def sample_approx(self, sampling_mode="AED", nb_iter=10, T_max=None):

		self.list_of_samples.append(sampl)

	def plot(self):
		"""Display a heatmap of the kernel provided, either K- or L-ensemble"""

		fig, ax = plt.subplots(1,1)

		heatmap = ax.pcolor(self.K, cmap='jet')

		ax.set_aspect('equal')

		ticks = np.arange(self.nb_items)
		ticks_label = [r'${}$'.format(tic) for tic in ticks]

		ax.xaxis.tick_top()
		ax.set_xticks(ticks+0.5, minor=False)

		ax.invert_yaxis()
		ax.set_yticks(ticks+0.5, minor=False)

		ax.set_xticklabels(ticks_label, minor=False)
		ax.set_yticklabels(ticks_label, minor=False)

		plt.colorbar(heatmap)
		plt.show()















class Discrete_kDPP(Discrete_DPP):
	"""docstring for DiscretekDPP"""
	def __init__(self, size, kernel, ensemble_type, projection_kernel=False):

		super().__init__(kernel, ensemble_type, projection_kernel)

		self.size = size
		self.__check_size_validity()
		self.el_sym_pol_eval = None

	def __str__(self):
		return self._str_info(self.size)

	def __check_size_validity(self):
		if (self.size <= 0) & (not isinstance(self.size, int)):
			raise ValueError("Invalid size parameter: must be a positive integer.\nGiven size = {}".format(self.size))

	def info(self):
		print(self.__str__())
		
	### Exact sampling
	def sample_exact(self, sampling_mode="GS"):

		self.sampling_mode = sampling_mode

		if self.eigendecomposition_available: # Use it!
			sampl = k_dpp_sampler_eig(self.eig_vals_L, 
															self.eig_vecs,
															self.size, 
															self.sampling_mode)
			
		elif (self.ensemble_type == 'K') & (self.projection_kernel): 
			# No need for eigendecomposition, update conditional via Gram-Schmidt on columns (equiv on rows) of K
			sampl = proj_k_dpp_sampler_kernel(self.K,
																			self.size, 
																			self.sampling_mode)

		else:
			raise ValueError("WARNING sampling!!")

		self.list_of_samples.append(sampl)

	### Approximate sampling
	def sample_approx(self, sampling_mode="AED", nb_iter=10, T_max=None):

		self.list_of_samples.append(sampl)
