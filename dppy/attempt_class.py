from .random_matrices import *
import matplotlib.pyplot as plt





class ReferenceMeasure:

	def __init__(self, name, **params):

		self.name = name.lower()
		self.__check_ref_measure_validity()

		self.params = params
		self.__check_params_validity()

	def info(self):
		print("ReferenceMeasure = {}, "
			  "with parameters = {}".format(self.name, self.params))

	def __check_ref_measure_validity(self):

		supported_ref_meas = ("normal", "gaussian",
													"gamma", 
													"beta", 
													"unif_unit_circle")

		if self.name not in supported_ref_meas:
			str_list = ["- {}".format(meas) for meas in supported_ref_meas]
			raise ValueError("\n".join(["Supported reference measures:"] + str_list))

	def __check_params_validity(self):

			if self.name in ("normal", "gaussian"):
				if ("loc" not in self.params) | ("scale" not in self.params):
					raise ValueError("mean and standard deviation are passed via "
													"a dict with keys 'loc' and 'scale'.\n"
													"Given {}".format(self.params))
				else:
					np.random.normal(self.params["loc"], 
									 self.params["scale"])

			elif self.name == "gamma":
				if ("shape" not in self.params) | ("scale" not in self.params):
					raise ValueError("shape and scale parameters are passed via "
													"a dict with keys 'shape' and 'scale'.\n"
													"Given {}".format(self.params))
				else:
					np.random.gamma(self.params["shape"], 
									scale=self.params["scale"])

			elif self.name == "beta":
				if ('a' not in self.params) | ('b' not in self.params):
					raise ValueError("shape parameters are passed via " 
													"a dict with keys 'a' and 'b'.\n"
													"Given {}".format(self.params))
				else:
					np.random.beta(self.params["a"], 
								   self.params["b"])

			elif self.name == "unif_unit_circle":
				pass

class BetaOPE:
	"""docstring for OPE"""

	def __init__(self, ref_meas, beta):

		self.ref_meas = ref_meas

		self.beta = beta
		self.__check_beta_validity()

		self.params = {}
		self.list_of_samples = []

	def info(self):
		self.ref_meas.info()
		info_str = "\n".join(["beta coefficient = {}",
												"Current number of samples = {}"])

		print(info_str.format(self.beta,
													len(self.list_of_samples)))

	def sample(self, nb_points):
		
		if self.ref_meas.name in ("normal", "gaussian"):
			sampl = muref_normal_sampler_tridiag(loc=self.ref_meas.params["loc"], 
																					scale=self.ref_meas.params["scale"], 
																					beta=self.beta, 
																					size=nb_points)

		elif self.ref_meas.name == "gamma":
			sampl = mu_ref_gamma_sampler_tridiag(shape=self.ref_meas.params["shape"],
																					scale=self.ref_meas.params["scale"],
																					beta=self.beta, 
																					size=nb_points)					

		elif self.ref_meas.name == "beta":
			sampl = mu_ref_beta_sampler_tridiag(a=self.ref_meas.params["a"], 
																				b=self.ref_meas.params["b"], 
																				beta=self.beta, 
																				size=nb_points)

		elif self.ref_meas.name == "unif_unit_circle":
			sampl = mu_ref_unif_unit_circle_sampler_quindiag(beta=self.beta, 
																  										size=nb_points)

		self.list_of_samples.append(sampl)

	def flush_samples(self):
		self.list_of_samples = []

	def kernel(self, list_of_points):
		# return the matrix [K(x,y)]_x,y in list_of_points
		# maybe plot the heatmap
		if self.beta != 2:
			raise ValueError("beta parameter {} != 2, the OPE is not a DPP"
							 				"there is no notion of kernel".format(self.beta))

	def __check_beta_validity(self):
		if not (self.beta > 0):
			raise ValueError("beta must be positive")

		if self.ref_meas.name == "unif_unit_circle":
			if not isinstance(self.beta, int):
				raise ValueError("Invalid beta parameter. "
												"beta must be positive integer.\n"
												"Given beta={}".format(self.beta))





















class ClassicalOPE:
	"""docstring for OPE"""

	def __init__(self, name, beta=2):

		self.name = name.lower()
		self.__check_name_validity()

		self.beta = beta
		self.__check_beta_validity()

		self.params = {}
		self.list_of_samples = []

	def info(self):
		info_str = "\n".join(["Classical OPE name = {}",
													"beta coefficient = {}",
													"params = {}",
													"Current number of samples = {}"])

		return print(info_str.format(self.name, 
																self.beta,
																self.params,
																len(self.list_of_samples)))

	def sample(self, matrix_model="banded", **params):

		self.params = params
		self.__check_params_validity()

		if matrix_model == "banded":

			if self.name == "hermite":
				sampl = hermite_sampler_tridiag(N=self.params['N'],
													 beta=self.beta)

			elif self.name == "laguerre":
				sampl = laguerre_sampler_tridiag(M=self.params['M'],
																				N=self.params['N'],
																				beta=self.beta)

			elif self.name == "jacobi":
				sampl = jacobi_sampler_tridiag(M_1=self.params['M_1'],
																			M_2=self.params['M_2'],
																			N=self.params['N'],
																			beta=self.beta)

			elif self.name == "circular":
				sampl = mu_ref_unif_unit_circle_sampler_quindiag(beta=self.beta, 
																												size=self.params['N'])

			elif self.name == "ginibre":
				raise ValueError("In valid 'matrix_model' argument, there is"
												"no banded model for the Ginibre ensemble. "
												"Use 'matrix_model'='full'")

		elif matrix_model == "full":

			if self.name == "hermite":
				sampl = hermite_sampler_full(N=self.params['N'],
																		beta=self.beta)

			elif self.name == "laguerre":
				sampl = laguerre_sampler_full(M=self.params['M'],
																			N=self.params['N'],
																			beta=self.beta)

			elif self.name == "jacobi":
				sampl = jacobi_sampler_full(M_1=self.params['M_1'],
																		M_2=self.params['M_2'],
																		N=self.params['N'],
																		beta=self.beta)

			elif self.name == "circular":
				sampl = circular_sampler_full(N=self.params['N'],
																			beta=self.beta,
																			mode=self.params["mode"])

			elif self.name == "ginibre":
				sampl = ginibre_sampler_full(N=self.params['N'])

		else:
			raise ValueError("Invalid matrix_model argument: "
											"'banded'(default) or 'full'.\n"
											"Given {}".format(matrix_model))

		self.list_of_samples.append(sampl)

	def flush_samples(self):
		self.list_of_samples = []

	def plot(self, normalization=True):

		if not self.list_of_samples:
			raise ValueError("List of samples is empty, you must sample first")

		fig, ax = plt.subplots(1, 1)
		points = self.list_of_samples[-1].copy()

		if not normalization:

			if (self.name == "circular") | (self.name == "ginibre"):
				if self.name == "circular":
					unit_circle = plt.Circle((0,0), 1, color='r', fill=False) 
					ax.add_artist(unit_circle) 


				ax.scatter(points.real, points.imag, c='blue', label="sample")

				ax.set_xlim([-1.3, 1.3])
				ax.set_ylim([-1.3, 1.3])

				ax.set_aspect('equal')

			else:
				ax.scatter(points, np.zeros(len(points)), c='blue', label="sample")

		else:

			if self.name == "hermite":
				points/= np.sqrt(self.beta * self.params['N'])
				ax.scatter(points, np.zeros(len(points)), c='blue', label="sample")

				x=np.linspace(-2,2,100)
				ax.plot(x, semi_circle_law(x), 
						'r-', lw=2, alpha=0.6, 
						label=r'$f_{sc}$')

			elif self.name == "laguerre":
				points /= self.beta * self.params['M']
				ax.scatter(points, np.zeros(len(points)), c='blue', label="sample")

				x=np.linspace(1e-3,3.5,100)
				ax.plot(x, marcenko_pastur_law(x,
																			self.params['M'], 
																			self.params['N']),
								'r-', lw=2, alpha=0.6,
								label=r'$f_{MP}$')

			elif self.name == "jacobi":				
				ax.scatter(points, np.zeros(len(points)), c='blue', label="sample")

				x=np.linspace(1e-5,1-1e-3,100)
				ax.plot(x, wachter_law(x,
															self.params['M_1'], 
															self.params['M_2'], 
															self.params['N']),
								'r-', lw=2, alpha=0.6, 
								label='Wachter Law')

			elif (self.name == "circular") | (self.name == "ginibre"):
				unit_circle = plt.Circle((0,0), 1, color='r', fill=False) 
				ax.add_artist(unit_circle) 

				if self.name == "ginibre":
					points /= np.sqrt(self.params['N'])

				ax.scatter(points.real, points.imag, c='blue', label="sample")

				ax.set_xlim([-1.3, 1.3])
				ax.set_ylim([-1.3, 1.3])

				ax.set_aspect('equal')

		ax.legend(loc='best', frameon=False)
		plt.show()

	def hist(self, normalization=True):
		if not self.list_of_samples:
			raise ValueError("Empty list of samples, you must sample first!")

		if self.name == "ginibre":
			raise ValueError("No 'hist' method for Ginibre.")

		fig, ax = plt.subplots(1, 1)
		points = np.array(self.list_of_samples).flatten()

		if self.name == "circular":
			ax.hist(np.angle(points), 
							bins=50, 
							density=1, 
							facecolor='blue', alpha=0.5, 
							label='hist')
			ax.axhline(y=1/(2*np.pi), color='r', label=r"$\frac{1}{2\pi}$")
		else:
			if not normalization:
				ax.hist(points, 
								bins=50, 
								density=1, 
								facecolor='blue', alpha=0.5, 
								label='hist')

			else:
				if self.name == "hermite":
					points /= np.sqrt(self.beta * self.params['N'])
					ax.hist(points, 
									bins=50, 
									density=1, 
									facecolor='blue', alpha=0.5, 
									label='hist')

					x=np.linspace(-2,2,100)
					ax.plot(x, semi_circle_law(x), 
									'r-', lw=2, alpha=0.6, 
									label=r'$f_{sc}$')

				elif self.name == "laguerre":
					points /= self.beta * self.params['M']
					ax.hist(points, 
									bins=50, 
									density=1, 
									facecolor='blue', alpha=0.5, 
									label='hist')
					
					x=np.linspace(1e-3,3.5,100)
					ax.plot(x, marcenko_pastur_law(x,
																				self.params['M'], 
																				self.params['N']),
									'r-', lw=2, alpha=0.6,
									label=r'$f_{MP}$')

				elif self.name == "jacobi":
					ax.hist(points, 
									bins=50, 
									density=1, 
									facecolor='blue', alpha=0.5, 
									label='hist')

					x=np.linspace(1e-5,1-1e-3,100)
					ax.plot(x, wachter_law(x,
																self.params['M_1'], 
																self.params['M_2'], 
																self.params['N']),
									'r-', lw=2, alpha=0.6, 
									label='Wachter Law')

		ax.legend(loc='best', frameon=False)
		plt.show()
		
	def kernel(self, list_of_points):
		# return the matrix [K(x,y)]_x,y in list_of_points
		# maybe plot the heatmap
		if self.beta != 2:
			raise ValueError("Invalid beta parameter, {} != 2. " 
											"The OPE is not a DPP, "
											"there is no notion of kernel".format(self.beta))
		else:
			pass

	def __check_name_validity(self):

		supported_classical_OPE = ("hermite, laguerre, jacobi,\
															circular,\
															ginibre")

		if self.name not in supported_classical_OPE:
			str_list = ["- {}".format(OPE) for OPE in supported_classical_OPE]
			raise ValueError("\n".join(["Supported OPEs:"] + str_list))

	def __check_beta_validity(self):
		supported_beta = (1, 2, 4)

		if self.name == "ginibre":
			if self.beta !=2:
				raise ValueError("Invalid beta parameter, only beta=2 available.\n"
												"Given {}".format(self.beta))
		if self.beta not in supported_beta:
			raise ValueError("Invalid beta parameter, "
											"must be equal to 1, 2 or 4.")

	def __check_params_validity(self):

		if self.name == "hermite":
			if "N" not in self.params:
				raise ValueError("beta(=1,2,4) hermite ensemble has one sampling"
												"parameter passed via a dict with key 'N'.\n"
												"Given {}".format(self.params))

		elif self.name == "laguerre":
			if ('M' not in self.params) |\
			   ('N' not in self.params):
				raise ValueError("beta(=1,2,4) laguerre ensemble has two sampling"
												"parameters passed via a dict with keys "
												"'M' and 'N', with M>=N.\n"
												"Given {}".format(self.params))

			elif self.params['M'] < self.params['N']:
				raise ValueError("beta(=1,2,4) laguerre ensemble has two sampling"
												"parameters that must satisfy M>=N.\n"
												"Given {}".format(self.params))

		elif self.name == "jacobi":
			if ('M_1' not in self.params) |\
			   ('M_2' not in self.params) |\
			   ('N' not in self.params):
				raise ValueError("beta(=1,2,4) jacobi ensemble has two sampling"
												"parameters passed via a dict with keys "
												"'M_1', 'M_2' and 'N', with M_1, M_2>=N.\n"
												"Given {}".format(self.params))

			elif (self.params['M_1'] < self.params['N']) |\
					(self.params['M_2'] < self.params['N']):
				raise ValueError("beta(=1,2,4) jacobi ensemble has two sampling"
												"parameters that must satisfy M_1,2 >= N.\n"
												"Given {}".format(self.params))

		elif self.name == "circular":
			if "mode" not in self.params:
				self.params["mode"] = "hermite"
			if "N" not in self.params:
				raise ValueError("beta(=1,2,4) circular ensemble has two sampling"
												"parameters passed via a dict with keys "
												"'N' and 'mode'(default='hermite' or 'QR').\n"
												"Given {}".format(self.params))

		elif self.name == "ginibre":
			if "N" not in self.params:
				raise ValueError("ginibre ensemble has one sampling"
												"parameter passed via a dict with key 'N'.\n"
												"Given {}".format(self.params))