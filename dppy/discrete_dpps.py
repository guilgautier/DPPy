# coding: utf-8
from .exact_sampling import *
from .approximate_sampling import *
import matplotlib.pyplot as plt

class Discrete_DPP:

	def __init__(self, kernel, ensemble_type, projection_kernel=False):

		self.nb_items = kernel.shape[0]
		self.projection_kernel = projection_kernel
		self.__check_projection_kernel(kernel)

		self.K = None
		self.L = None
		self.ensemble_type = ensemble_type
		self.__check_ensemble_type(kernel)

		self.list_of_samples = []
		self.sampling_mode = None # Default 'GS'
		### Exact sampling: eigendecomposition may be required
		# 'GS' for Gram-Schmidt, 
		# 'Schur' for Schur complement 
		# 'KuTa12' for Kulesza (Algo 1).
		self.eigen_decomposition_available = False
		self.eig_vals_K = None
		self.eig_vals_L = None
		self.eig_vecs = None
		self.__check_kernel_for_dpp_validity(kernel) 

		### Approx sampling: eigendecomposition not required
		# 'AED' Add-Exchange-Delete
		# 'AD' Add-Delete
		# 'E' Exchange


	def __str__(self):
		str_info = ["Discrete DPP defined by:", 
								"- {}-ensemble on {} items",
								"- Projection kernel: {}",
								"- sampling mode = {}",
								"- number of samples = {}"]

		return "\n".join(str_info).format(self.ensemble_type, self.nb_items,
																		"Yes" if self.projection_kernel else "No",
																		self.sampling_mode,
																		len(self.list_of_samples))

	def __check_projection_kernel(self):
		if not isinstance(self.projection_kernel, bool):
			str_list = ["Invalid projection_kernel argument: must be True/False",
									"Given projection_kernel={}".format(self.projection_kernel)]
			raise ValueError("\n".join(str_list))

		elif self.projection_kernel:
			# Cheap test checking reproducing property
			nb_tmp = 5
			items_to_check = np.arange(nb_tmp)
			K_i_ = kernel[items_to_check, :]
			K_ii = kernel[items_to_check, items_to_check]

			if not np.allclose(np_inner1d(K_i_, K_i_), K_ii):
				raise ValueError("Invalid kernel: doesn't seem to be a projection")

	def __check_ensemble_type(self, kernel):

		if self.ensemble_type == 'K':
			self.K = kernel
		elif self.ensemble_type == 'L':
			self.L = kernel
		else:
			str_list = ["Invalid ensemble_type parameter, use:",
									"- 'K' for inclusion probability kernel",
									"- 'L' for marginal kernel.",
									"Given {}".format(self.ensemble_type)]

			raise ValueError("\n".join(str_list))

	def __eigendecompose(self):
		"""Check symmetry, projection, and validity:
		- For K-ensemble 0<=K<=I
		- For L-ensemble L>=0"""
		if self.eigen_decomposition_available:
			pass
		else:
			# Eigendecomposition necessary for non projection kernel
			tol = 1e-8 # tolerance on the eigenvalues
			self.eigendecomposition = True

			# If K-ensemble
			if self.ensemble_type == 'K':
				self.eig_vals_K, self.eig_vecs = la.eigh(self.K)
				# Check 0 <= K <= I
				if not np.all((-tol<=self.eig_vals_K) & (self.eig_vals_K<=1.0+tol)):
					raise ValueError("Invalid kernel for K-ensemble. Eigen values are not in [0,1]")

			# If L-ensemble
			elif self.ensemble_type == 'L':
				self.eig_vals_L, self.eig_vecs = la.eigh(self.L)
				# Check L >= 0
				if not np.all(eig_vals >= -tol):
					raise ValueError("Invalid kernel for L-ensemble. Eigen values !>= 0")

	def info(self):
		print(self.__str__())

	def compute_K_kernel(self):
		"""K = L(I+L)^-1 = I - (I+L)^-1"""
		if self.K is not None:
			raise ValueError("'K' kernel is already available")

		elif self.eigen_decomposition_available:
			self.eig_vals_K = self.eig_vals_L/(1.0 + self.eig_vals_L)
			self.K = (self.eig_vecs * self.eig_vals_K).dot(self.eig_vecs.T)

		else:
			print("Eigendecomposition performed")
			self.__eigendecompose()
			self.compute_K_kernel()

	def compute_L_kernel(self):
		"""L = K(I-K)^-1 = (I-K)^-1 - I"""
		if self.L is not None:
			raise ValueError("'L' kernel is already available")

		elif self.eigen_decomposition_available:
			try: # to compute eigenvalues of kernel L = K(I-K)^-1
				np.seterr(divide='raise')
				self.eig_vals_L = self.eig_vals_K/(1.0 - self.eig_vals_K)
				self.L = (self.eig_vecs * self.eig_vals_L).dot(self.eig_vecs.T)

			except FloatingPointError as e:
				str_list = ["WARNING: {}.".format(e),
										"Eigenvalues of 'L' kernel (L=K(I-K)^-1) cannot be computed.",
										"'K' kernel has some eigenvalues equal are very close to 1.",
										"Hint: 'K' kernel might be a projection."]
				print("\n".join(str_list))

		else:
			print("Eigendecomposition performed")
			self.__eigendecompose()
			self.compute_L_kernel()

	def flush_samples(self):
		self.list_of_samples = []

	### Exact sampling
	def sample_exact(self, sampling_mode="GS"):

		self.sampling_mode = sampling_mode

		if (self.ensemble_type == 'K') & (self.projection_kernel): 
			# If K is orthogonal projection no need for eigendecomposition, update conditional via Gram-Schmidt on columns (equiv on rows) of K
			sampl = proj_dpp_sampler_kernel(self.K, 
																			self.sampling_mode)

		else:
			if self.eig_vals_K is None: # Compute eig_vals_K from eig_vals_L
				self.eig_vals_K = self.eig_vals_L/(1.0 + self.eig_vals_L)

			sampl = dpp_sampler_eig(self.eig_vals_K, 
															self.eig_vecs, 
															self.sampling_mode)

		self.list_of_samples.append(sampl)

	### Approximate sampling
	def sample_approx(self, sampling_mode="AED", nb_iter=10, T_max=None):

		self.list_of_samples.append(sampl)

	def plot(self):
		"""Display a heatmap of the kernel provided, either K- or L-ensemble"""

		print("Heat map of '{}'-kernel".format('K' if self.ensemble_type == 'K'\
																							else 'L'))
		fig, ax = plt.subplots(1,1)

		heatmap = ax.pcolor(self.K if self.ensemble_type == 'K' else self.L, 
												cmap='jet')

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