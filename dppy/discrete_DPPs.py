# coding: utf-8
from exact_sampling import *

class DiscreteDPP:

	def __init__(self, kernel, ensemble_type='K', projection_kernel=False):

		self.K = kernel
		self.ensemble_type = ensemble_type
		self.projection_kernel = projection_kernel

		self.ber_params_sampling = None
		# for projection = eig_vals
		# for non projection = eig_vals/(1+eigvals)
		self.eig_vecs = None
		self.__check_kernel_validity()

		self.nb_items = self.K.shape[0]

		self.sampling_mode = None
		# self.sampling_params = {}
		self.list_of_samples = []

	def __check_kernel_validity(self):

		# Check symmetry of the kernel
	    if np.array_equal(self.K, self.K.T):

	    	if projection_kernel: 
	    	# Perform cheap test of the reproducing property
	    		nb_tmp = 3
	    		items_to_check = np.arange(nb_tmp)
	    		K_i_ = self.K[items_to_check, :]
	    		K_ii = self.K[items_to_check, items_to_check]

				if not np.allclose(inner1d(K_i_, K_i_), K_ii):
					raise ValueError("Kernel doesn't seem to be a projection")

	    	else: # If kernel is not assumed to be a projection, compute eigendecomposition
				eig_vals, self.eig_vecs = la.eigh(self.K)
		
				# If K-ensemble
				if self.ensemble_type == 'K': 
					# Check 0 <= K <=1
					if not np.all((0.0<=eig_vals) & (eig_vals<=1.0)):
						raise ValueError("Invalid kernel for {} ensemble. Eigen values are not in [0,1]".format(self.ensemble_type))
					else:
						self.ber_params_sampling = eig_vals

				# If L-ensemble
				else self.ensemble_type == 'L':
					# Check L >= 0
					if not np.all(0.0<=eig_vals):
						raise ValueError("Invalid kernel for {} ensemble. Eigen values are not >= 0".format(self.ensemble_type))
					else:
						self.ber_params_sampling = eig_vals/(1.0+eig_vals)

	    else:
	    	raise ValueError("Invalid kernel: not symmetric")


	def info(self):
		str_info = ["Discrete DPP defined via a {} ensemble on {}",\
					"Projection kernel: {}"
					"sampling mode = {}.",\
					"number of samples = {}."]

		print("\n".join(str_info).format(self.ensemble_type,
									self.nb_items,
									"Yes" if self.projection_kernel else "No",
									self.sampling_mode,
									len(self.list_of_samples)))

	def sample(self, sampling_mode="GS"):

		if self.sampling_mode == "GS":# Gram Schmidt update

			if self.projection_kernel:
				sampl = projection_dpp_sampler_GS(self.K)
			else:
				sampl = dpp_sampler_eig_GS(self.ber_params_sampling, 
										self.eig_vecs)

		if self.sampling_mode == "Schur":# Schur complement update

			if self.projection_kernel:
				sampl = projection_dpp_sampler_Schur(self.K)
			else:
				raise ValueError("sampling_mode='Schur' is not available for non projection kernel.\nChoose 'GS' or 'KuTa12'.")

		if self.sampling_mode == "KuTa12":

			if self.projection_kernel:
				self.ber_params_sampling, self.eig_vecs = la.eigh(self.K)
				sampl = dpp_sampler_KuTa12(self.ber_params_sampling, 
										self.eig_vecs)
			else:
				sampl = dpp_sampler_KuTa12(self.ber_params_sampling, 
										self.eig_vecs)
		else:
			raise ValueError("Invalid sampling_mode parameter, choose among:\n\
				- 'GS' (default),\n\
				- 'Schur'")