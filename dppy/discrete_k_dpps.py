# coding: utf-8
try: # Local import
	from .exact_sampling import *
	from .approximate_sampling import *
except (SystemError, ImportError):
	from .exact_sampling import *
	from .approximate_sampling import *
import matplotlib.pyplot as plt

class Discrete_k_DPP:

	def __init__(self, kernel, size, projection_kernel=False):

		self.nb_items = kernel.shape[0]

		self.size = size
		self.__check_size_param_validity()

		self.L = kernel
		self.projection_kernel = projection_kernel
		self.__check_projection_kernel_param_validity()

		# If valid kernel, diagonalization only for non projection kernel.
		self.eig_vals = None
		self.eig_vecs = None
		self.__check_kernel_for_k_dpp_validity(kernel)

		if self.projection_kernel:
			self.el_sym_pol_eval = None
		else:
			self.el_sym_pol_eval = elem_symm_poly(self.eig_vals, self.size)

		self.sampling_mode = None # Default 'GS'
		### Exact sampling
		# 'GS' for Gram-Schmidt,
		# 'Schur' for Schur complement
		# 'KuTa12' for Kulesza (Algo 1).
		### Approx sampling
		# 'AED' 'AD' 'E' A=Add E=Exchange D=Delete
		self.list_of_samples = []

	def __str__(self):
		str_info = ["Discrete k-DPP(L) defined on {} items by:",
								"- Projection kernel: {}",
								"- sampling mode = {}",
								"- number of samples = {}"]

		return "\n".join(str_info).format(self.nb_items,
																		"Yes" if self.projection_kernel else "No",
																		self.sampling_mode,
																		len(self.list_of_samples))

	def __check_size_param_validity(self):
		if (self.size <= 0) & (not isinstance(self.size, int)):
			raise ValueError("Invalid size parameter: must be a positive integer.\nGiven size = {}".format(self.size))

	def __check_projection_kernel_param_validity(self):
		# Recall we first checked the kernel to be symmetric (K=K.T), here we check the reproducing property of the (orthogonal) projection kernel.
		# For this we perform a cheap test to check K^2 = K K.T = K
		if not isinstance(self.projection_kernel, bool):
			str_list = ["Invalid projection_kernel argument: must be True/False",
									"Given projection_kernel={}".format(self.projection_kernel)]
			raise ValueError("\n".join(str_list))

	def __check_kernel_for_k_dpp_validity(self, kernel):
		"""Check symmetry, projection, and validity:
		- For inclusion kernel 0<=K<=I
		- For marginal kernel L>=0"""

		# Check symmetry: L.T = L
		if not np.allclose(self.L, self.L.T):
			raise ValueError("Invalid kernel: not symmetric")

		# Check if 'L' kernel is orthogonal projection L^2 = L L.T = L
		# Only case for which eigendecomposition is not necessary
		if self.projection_kernel:
			# Cheap test checking reproducing property
			nb_tmp = 5
			items_to_check = np.arange(nb_tmp)
			L_i_ = kernel[items_to_check, :]
			L_ii = kernel[items_to_check, items_to_check]

			if np.allclose(np_inner1d(L_i_, L_i_), L_ii):
					self.L = kernel
			else:
				raise ValueError("Invalid kernel: doesn't seem to be a projection")

		else:
			# Eigendecomposition necessary for non projection kernel
			eig_vals, eig_vecs = la.eigh(kernel)
			tol = 1e-8 # tolerance on the eigenvalues

			# Check L >= 0
			if np.all(eig_vals >= -tol):
				self.L = kernel
				self.eig_vals = eig_vals
				self.eig_vecs = eig_vecs
			else:
				raise ValueError("Invalid kernel for marginal kernel. Eigen values !>= 0")

	def info(self):
		print(self.__str__())

	def flush_samples(self):
		self.list_of_samples = []

	### Exact sampling
	def sample_exact(self, sampling_mode="GS"):

		self.sampling_mode = sampling_mode
		if self.projection_kernel:
			# No need for eigendecomposition, update conditional via Gram-Schmidt on columns (equiv on rows) of K
			sampl = proj_k_dpp_sampler_kernel(self.L,
																				self.size,
																				self.sampling_mode)
		else: #if self.el_sym_pol_eval is not None:
		# i.e. if eigen decomposition available use it!
			sampl = k_dpp_sampler_eig(self.eig_vals, self.eig_vecs,	self.size,
																self.sampling_mode,
																self.el_sym_pol_eval)

		self.list_of_samples.append(sampl)

	# ### Approximate sampling
	# def sample_approx(self, sampling_mode="AED", nb_iter=10, T_max=None):

	# 	self.list_of_samples.append(sampl)

	# def plot(self):
	# 	"""Display a heatmap of the kernel provided, either K- or marginal kernel"""

	# 	print("Heat map of L-kernel")
	# 	fig, ax = plt.subplots(1,1)

	# 	heatmap = ax.pcolor(self.L, cmap='jet')

	# 	ax.set_aspect('equal')

	# 	ticks = np.arange(self.nb_items)
	# 	ticks_label = [r'${}$'.format(tic) for tic in ticks]

	# 	ax.xaxis.tick_top()
	# 	ax.set_xticks(ticks+0.5, minor=False)

	# 	ax.invert_yaxis()
	# 	ax.set_yticks(ticks+0.5, minor=False)

	# 	ax.set_xticklabels(ticks_label, minor=False)
	# 	ax.set_yticklabels(ticks_label, minor=False)

	# 	plt.colorbar(heatmap)
	# 	plt.show()