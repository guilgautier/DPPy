# coding: utf-8
from .exact_sampling import *
import matplotlib.pyplot as plt

class DiscreteDPP:

	def __init__(self, kernel, ensemble_type, projection_kernel=False):

		self.K = kernel
		self.nb_items = self.K.shape[0]
		self.__check_kernel_symmetry()

		self.ensemble_type = ensemble_type
		self.__check_ensemble_type_validity()

		self.projection_kernel = projection_kernel
		self.__check_projection_kernel_validity()

		self.__eigen_decomposition_available = False
		self.eig_vals = None
		self.eig_vecs = None
		self.__check_kernel_for_dpp_validity() # If valid, diagonalization is performed only for non projection kernel.

		self.ber_params_sampling = np.zeros(self.nb_items)
		# for K-ensemble = eig_vals
		# for L-ensemble = eig_vals/(1+eigvals)
		self.sampling_mode = "GS" # 'GS' for Gram-Schmidt, 'Schur' for Schur complement and 'KuTa12' for Kulesza (Algo 1).
		self.list_of_samples = []

	def __check_ensemble_type_validity(self):

		if self.ensemble_type not in ('K', 'L'):
			str_list = ["Invalid ensemble_type parameter, use:",
									"- 'K' for inclusion probability kernel",
									"- 'L' for marginal kernel.",
									"Given: ensemble_type = '{}'"]

			raise ValueError("\n".join(str_list).format(self.ensemble_type))

	def __check_kernel_symmetry(self):

		# Check symmetry of the kernel K.T = K
		if not np.allclose(self.K, self.K.T):
			raise ValueError("Invalid kernel: not symmetric")

	def __check_projection_kernel_validity(self):
		# Recall we first checked the kernel to be symmetric (K=K.T), here we check the reproducing property of the (orthogonal) projection kernel.
		# For this we perform a cheap test K^2 = K K.T = K
		if self.projection_kernel:
			nb_tmp = 3
			items_to_check = np.arange(nb_tmp)
			K_i_ = self.K[items_to_check, :]
			K_ii = self.K[items_to_check, items_to_check]

			if not np.allclose(inner1d(K_i_, K_i_), K_ii):

				raise ValueError("Kernel doesn't seem to be a projection")

	def __check_kernel_for_dpp_validity(self):

		if not self.projection_kernel: 
		# Compute eigendecomposition of the (symmetric) kernel
			self.eigen_decompose()

			# If K-ensemble
			if self.ensemble_type == 'K': 
				# Check 0 <= K <= I
				if not np.all((0.0<=self.eig_vals) & (self.eig_vals<=1.0)):
					raise ValueError("Invalid kernel for K-ensemble. Eigen values are not in [0,1]")

			# If L-ensemble
			elif self.ensemble_type == 'L':
				# Check L >= 0
				if not np.all(self.eig_vals>=0.0):
					raise ValueError("Invalid kernel for L-ensemble. Eigen values are not >= 0")


	# def __compute_bernouilli_parameters_for_sampling(self):

	# 		if self.ensemble_type == 'K':
	# 			self.ber_params_sampling = self.eig_vals

	# 		elif self.ensemble_type == 'L':
	# 			self.ber_params_sampling = self.eig_vals/(1.0+self.eig_vals)

	def info(self):
		str_info = ["Discrete DPP defined via a {}-ensemble on {} items",
								"Projection kernel: {}",
								"sampling mode = {}",
								"number of samples = {}"]

		print("\n".join(str_info).format(self.ensemble_type, self.nb_items,
																		"Yes" if self.projection_kernel else "No",
																		self.sampling_mode,
																		len(self.list_of_samples)))

	def flush_samples(self):
		self.list_of_samples = []

	def eigen_decompose(self):

		if not self.__eigen_decomposition_available:
			self.eig_vals, self.eig_vecs = la.eigh(self.K)
			self.__eigen_decomposition_available = True

			if self.ensemble_type == 'K':
				self.ber_params_sampling = self.eig_vals

			elif self.ensemble_type == 'L':
				self.ber_params_sampling = self.eig_vals/(1.0+self.eig_vals)


		else:
			print("Eigendecomposition already available")

	def sample(self, sampling_mode="GS"):

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

			if self.projection_kernel:
				self.eigen_decompose()

			sampl = dpp_sampler_KuTa12(self.ber_params_sampling, self.eig_vecs)

		else:
			str_list = ["Invalid sampling_mode parameter, choose among:",
									"- 'GS' (default)",
									"- 'Schur'"]
			raise ValueError("\n".join(str_list))

		self.list_of_samples.append(sampl)

	def plot(self):

		fig, ax = plt.subplots(1,1)

		heatmap = ax.pcolor(self.K, cmap='jet')

		ax.set_aspect('equal')

		ticks = np.arange(self.nb_items)
		ticks_label = [r'${}$'.format(i) for i in range(self.nb_items)]

		ax.xaxis.tick_top()
		ax.set_xticks(ticks+0.5, minor=False)

		ax.invert_yaxis()
		ax.set_yticks(ticks+0.5, minor=False)

		ax.set_xticklabels(ticks_label, minor=False)
		ax.set_yticklabels(ticks_label, minor=False)

		plt.colorbar(heatmap)
		plt.show()