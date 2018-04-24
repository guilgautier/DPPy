from .random_matrices import *

class ReferenceMeasure:

	def __init__(self, name, **params):

		self.name = name.lower()
		self.__test_ref_measure_validity()

		self.params = params
		self.__test_params_validity()

	def __test_ref_measure_validity(self):

		supported_ref_meas = ("normal", "gaussian",
							   "gamma", 
							   "beta", 
							   "unif_unit_circle")

		if self.name not in supported_ref_meas:
			str_list = ["- {}".format(meas) for meas in supported_ref_meas]
			raise ValueError("\n".join(["Supported reference measures:"] + str_list))

	def __test_params_validity(self):

			if self.name in ("normal", "gaussian"):
				if ("loc" not in self.params) | ("scale" not in self.params):
					raise ValueError("mean and standard deviation are passed via\
									  the dict 'params' with keys \
									  'loc' and 'scale'.\n\
									  Given {}".format(self.params))
				else:
					np.random.normal(self.params["loc"], 
									 self.params["scale"])

			elif self.name == "gamma":
				if ("shape" not in self.params) | ("scale" not in self.params):
					raise ValueError("shape and scale parameters are passed via\
									  the dict 'params' with keys \
									  'shape' and 'scale'.\n\
									  Given {}".format(self.params))
				else:
					np.random.gamma(self.params["shape"], 
									scale=self.params["scale"])

			elif self.name == "beta":
				if ('a' not in self.params) | ('b' not in self.params):
					raise ValueError("shape parameters are passed via\
									  the dict 'params' with keys \
									  'a' and 'b'.\n\
									  Given {}".format(self.params))
				else:
					np.random.beta(self.params["a"], 
								   self.params["b"])

			elif self.name == "unif_unit_circle":
				pass

	def info(self):
		print("ReferenceMeasure = {}, with parameters = {}".format(self.name, 
																   self.params))


class BetaOPE:
	"""docstring for OPE"""

	def __init__(self, ref_meas, beta=None):

		self.ref_meas = ref_meas

		if beta is None:
			self.beta = 2
		else:
			self.beta = beta

	def info(self):
		self.ref_meas.info()

	def sample(self, nb_points, mode="banded_model"):
		
		if self.ref_meas.name in ("normal", "gaussian"):
			return muref_normal_sampler_tridiag_model(
						loc=self.ref_meas.params["loc"], 
						scale=self.ref_meas.params["scale"], 
						beta=self.beta, 
						size=nb_points)

		elif self.ref_meas.name == "gamma":
			return mu_ref_gamma_sampler_tridiag_model(
						shape=self.ref_meas.params["shape"], 
						scale=self.ref_meas.params["scale"], 
						beta=self.beta, 
						size=nb_points)					

		elif self.ref_meas.name == "beta":
			return mu_ref_beta_sampler_tridiag_model(
						a=self.ref_meas.params["a"], 
						b=self.ref_meas.params["b"], 
						beta=self.beta, 
						size=nb_points)

		elif self.ref_meas.name == "unif_unit_circle":
			return mu_ref_unif_unit_circle_sampler_quindiag_model(
						beta=self.beta, 
						size=nb_points)

	def kernel(list_of_points):
		# return the matrix [K(x,y)]_x,y in list_of_points
		# maybe plot the heatmap

		return	