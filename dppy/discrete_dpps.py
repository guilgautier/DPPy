# coding: utf-8
from .exact_sampling import *
import matplotlib.pyplot as plt

class Discrete_DPP:

	def __init__(self, kernel, ensemble_type, projection_kernel=False):

		self.ensemble_type = ensemble_type
		self.__check_ensemble_type_validity()

		self.projection_kernel = projection_kernel
		self.__check_projection_kernel_validity()

		self.K = None
		self.L = None

		# If valid kernel, diagonalization only for non projection kernel.
		self.__eigen_decomposition_available = False
		self.eig_vals = None
		self.eig_vecs = None

		self.ber_params_sampling = None
		# for K-ensemble = eig_vals
		# for L-ensemble = eig_vals/(1+eigvals)
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

		# Check orthogonal projection kernel^2 = kernel kernel.T = K
		if self.projection_kernel:
			nb_tmp = 3
			items_to_check = np.arange(nb_tmp)
			K_i_ = kernel[items_to_check, :]
			K_ii = kernel[items_to_check, items_to_check]

			if not np.allclose(np_inner1d(K_i_, K_i_), K_ii):
				raise ValueError("Invalid kernel: kernel doesn't seem to be a projection")

		else: # If not orthogonal projection kernel compute eigendecomposition
			self.__eigen_decompose(kernel)
			tol = 1e-8

			# If K-ensemble
			if self.ensemble_type == 'K': 
				# Check 0 <= K <= I
				if np.all((-tol <= self.eig_vals) & (self.eig_vals <= 1.0+tol)):
					self.K = kernel
					self.ber_params_sampling = self.eigvals
				else:
					raise ValueError("Invalid kernel for K-ensemble. Eigen values are not in [0,1]")

			# If L-ensemble
			elif self.ensemble_type == 'L':
				# Check L >= 0
				if np.all(self.eig_vals >= -tol):
					self.L = kernel
					self.ber_params_sampling = self.eigvals/(1.0+self.eigvals)
				else:
					raise ValueError("Invalid kernel for L-ensemble. Eigen values are not >= 0")

	def __eigen_decompose(self, kernel):

		self.eig_vals, self.eig_vecs = la.eigh(kernel)
		self.__eigen_decomposition_available = True

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

	def compute_K_kernel(self):
		"""K = L(I+L)^-1 = I - (I+L)^-1"""
		if self.L:
			if not self.__eigen_decomposition_available:
				self.eigen_decompose(self.L)
				tmp = self.eigvals/(1.0+self.eig_vals)

			self.K = (self.eig_vecs * tmp).dot(self.eig_vecs.T)

		else:
			raise ValueError("'L' kernel not available => cannot compute 'K' kernel")

	def compute_L_kernel(self):
		"""L = K(I-K)^-1 = (I-K)^-1 - I"""
		if self.K:
			if not self.__eigen_decomposition_available:
				self.eigen_decompose(self.K)
				tmp = self.eigvals/(1.0-self.eig_vals)

			self.L = (self.eig_vecs * tmp).dot(self.eig_vecs.T)

		else:
			raise ValueError("'K' kernel not available => cannot compute 'K' kernel")

	def flush_samples(self):
		self.list_of_samples = []

	def sample_exact(self, sampling_mode="GS"):

		if self.sampling_mode == "GS":# Gram Schmidt update

			if self.projection_kernel:
				sampl = projection_dpp_sampler_GS(self.K)

			else:
				sampl = dpp_sampler_eig_GS(self.ber_params_sampling, self.eig_vecs)

		elif self.sampling_mode == "Schur":# Schur complement update

			if self.projection_kernel:
				sampl = projection_dpp_sampler_Schur(self.K)

			else:
				raise ValueError("sampling_mode='Schur' is not available for non projection kernel.\nChoose 'GS' or 'KuTa12'.")

		elif self.sampling_mode == "KuTa12":

			if (self.projection_kernel) & (not self.__eigen_decomposition_available):
				self.eigen_decompose()

			sampl = dpp_sampler_KuTa12(self.ber_params_sampling, self.eig_vecs)

		else:
			str_list = ["Invalid sampling_mode parameter, choose among:",
									"- 'GS' (default)",
									"- 'Schur'"]
			raise ValueError("\n".join(str_list))

		self.list_of_samples.append(sampl)

	def sample_approx(self, sampling_mode="AED", nb_iter=10, T_max=None):

		if self.ensemble_type == 'K':
			if self.__eigen_decomposition_available:
				self.

		if self.sampling_mode == "AED":# Add-Exchange-Delete

			if not self.projection_kernel:
				sampl = add_exchange_delete_sampler(kernel, nb_it_max, T_max)

			else:
				raise ValueError("Invalid sampling_mode parameter for proje")

		elif self.sampling_mode == "AD":# Add-Delete

			sampl = add_delete_sampler(kernel, nb_it_max, T_max)

		elif self.sampling_mode == "E":# (Basis) Exchange

		else:
			str_list = ["Invalid sampling_mode parameter, choose among:",
									"- 'AED' (default) Add-Exchange-Delete",
									"- 'AD', Add-Delete",
									"- 'E', Exchange",
									"Given: sampling_mode = {}".format(sampling_mode)]
			raise ValueError("\n".join(str_list))

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

	def __str__(self):
		return self._str_info(self.size)

	def __check_size_validity(self):
		if (self.size <= 0) & (not isinstance(self.size, int)):
			raise ValueError("Invalid size parameter: must be a positive integer.\nGiven size = {}".format(self.size))

	def info(self):
		print(self.__str__())
		
	def sample_exact(self, sampling_mode="GS"):

		if self.sampling_mode == "GS":# Gram Schmidt update

			if self.projection_kernel:
				sampl = projection_dpp_sampler_GS(self.K, self.size)

			else:
				self.ber_params_sampling = select_eig_vec(self.eig_vals, self.size)
				sampl = dpp_sampler_eig_GS(self.ber_params_sampling, self.eig_vecs)

		elif self.sampling_mode == "Schur":# Schur complement update

			if self.projection_kernel:
				sampl = projection_dpp_sampler_Schur(self.K, self.size)

			else:
				raise ValueError("sampling_mode='Schur' is not available for non projection kernel.\nChoose 'GS' or 'KuTa12'.")

		elif self.sampling_mode == "KuTa12":

			if (self.projection_kernel) & (not self.__eigen_decomposition_available):
				self.eigen_decompose()

			self.ber_params_sampling = select_eig_vec(self.eig_vals, self.size)
			sampl = dpp_sampler_KuTa12(self.ber_params_sampling, self.eig_vecs)

		else:
			str_list = ["Invalid sampling_mode parameter, choose among:",
									"- 'GS' (default)",
									"- 'Schur'"]
			raise ValueError("\n".join(str_list))

		self.list_of_samples.append(sampl)