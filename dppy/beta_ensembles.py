# coding: utf-8
try: # Local import
	from .random_matrices import *
except (SystemError, ImportError):
	from random_matrices import *
import matplotlib.pyplot as plt

class BetaEnsemble:
	""" Finite DPP object parametrized by

	:param name:
		- ``'hermite'``
		- ``'laguerre'``
		- ``'jacobi'``
		- ``'circular'``
		- ``'ginibre'``
	:type name:
		string

	:param beta:
		:math:`\\beta > 0` inverse temperature parameter.
		Default ``beta=2`` corresponds to the DPP case, see :ref:`beta_ensembles_definition_OPE`
	:type beta:
		int, float, default 2

	.. seealso::

		- :math:`\\beta`-Ensembles :ref:`beta_ensembles_definition`
	"""

	def __init__(self, name, beta=2):

		self.name = name.lower()
		self.__check_name_validity()

		self.beta = beta
		self.__check_beta_validity()

		self.mode = "banded" if self.name != "ginibre" else None
		self.params = {}
		self.list_of_samples = []
		# self.nb_of_samples = len(self.list_of_samples)

	def __str__(self):
		str_info = ["ensemble name = {}.",
								"beta parameter = {}.",
								"sampling mode = {}.",
								"sampling parameters = {}.",
								"number of samples = {}."]

		return "\n".join(str_info).format(self.name,
																			self.beta,
																			self.mode,
																			self.params,
																			len(self.list_of_samples))

	def info(self):
		""" Print infos about the :class:`BetaEnsemble` object
		"""
		print(self.__str__())

	def flush_samples(self):
		""" Empty the ``BetaEnsemble.list_of_samples`` attribute.
		"""
		self.list_of_samples = []

	def sample(self, mode='banded', **params):
		""" Sample exactly from the corresponding :class:`BetaEnsemble <BetaEnsemble>` object by computing the eigenvalues of random matrices.

		:param mode:

			- ``'banded'``: tri/quindiagonal matrix model
			- ``'full'``: full matrix model

		:type mode:
			string, default ``'banded'``

		:param params:
			Dictionary containing the parametrization of the underlying :class:`BetaEnsemble <BetaEnsemble>` object viewed as the eigenvalues of a full or banded random matrix.

			For ``mode='full'``, the ``'N'`` key refers to the number of points i.e. the size of the matrix to be diagonalized.

				- for ``BetaEnsemble.name='hermite'``

					``params={'N':N}``

				- for ``BetaEnsemble.name='laguerre'``

					``params={'M':M, 'N':N}`` where :math:`M \geq N`

				- for ``BetaEnsemble.name='jacobi'``

					``params={'M_1': M_1, 'M_2':M_2, 'N':N}`` where :math:`M_{1,2}\geq N`

				- for ``BetaEnsemble.name='circular'``

					``params={'N':N, 'haar_mode':'QR'/'hermite'}``

				- for ``BetaEnsemble.name='ginibre'``

					``params={'N':N}``

			For ``mode='banded'``, the ``'size'`` key refers to the number of points i.e. the size of the matrix to be diagonalized.

				- for ``BetaEnsemble.name='hermite'``

					- ``params={'loc':, 'scale':, 'size':}``, where ``'loc', 'scale'`` are respectively the mean and standard deviation of the corresponding Gaussian reference measure. To recover the full matrix model take, ``loc`` :math:`=0`, ``scale`` :math:`=\sqrt{2}` and ``size``:math:`=N`.

				- for ``BetaEnsemble.name='laguerre'``

					- ``params={'shape':, 'scale':, 'size':}``, where ``'shape', 'scale'`` are respectively the shape and standard deviation of the corresponding Gamma reference measure. To recover the full matrix model take ``shape`` :math:`=\\frac{1}{2} \\beta (M-N+1)`, ``scale``:math:`=2` and ``size`` :math:`=N`.

				- for ``BetaEnsemble.name='jacobi'``

					- ``params={'a':, 'b':, 'size':}``, where ``'a', 'b'`` are the respective parameters of the corresponding Beta reference measure. To recover the full matrix model take :math:`a=\\frac{1}{2} \\beta (M_1-N+1)`, :math:`b=\\frac{1}{2} \\beta (M_2-N+1)` and ``size``:math:`=N`.

				- for ``BetaEnsemble.name='circular'``

					- ``params={size:}``.

		:type params:
			dict

		.. seealso::

			- :func:`flush_samples <flush_samples>`
			- :ref:`full_matrix_models` and :ref:`banded_matrix_models`
		"""

		self.mode = mode
		self.__check_mode_validity()

		self.params = params
		self.__check_params_validity()

		if self.mode == "banded":

			if self.name == "hermite":
					sampl = muref_normal_sampler_tridiag(loc=self.params["loc"],
																					scale=self.params["scale"],
																					beta=self.beta,
																					size=self.params["size"])

			elif self.name == "laguerre":
					sampl = mu_ref_gamma_sampler_tridiag(
																					shape=self.params["shape"],
																					scale=self.params["scale"],
																					beta=self.beta,
																					size=self.params["size"])

			elif self.name == "jacobi":
					sampl = mu_ref_beta_sampler_tridiag(a=self.params["a"],
																						b=self.params["b"],
																						beta=self.beta,
																						size=self.params["size"])

			elif self.name == "circular":
					sampl = mu_ref_unif_unit_circle_sampler_quindiag(beta=self.beta,
																		  			size=self.params["size"])

			elif self.name == "ginibre":

					raise ValueError("In valid 'mode' argument. No banded model for Ginibre ensemble. Use 'full'.\nGiven {}".format(mode))

		elif self.mode == "full":

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
																			haar_mode=self.params["haar_mode"])

			elif self.name == "ginibre":
				sampl = ginibre_sampler_full(N=self.params['N'])

		self.list_of_samples.append(sampl)


	def plot(self, normalization=True, title=""):
		""" Display the last realization of the corresponding :class:`BetaEnsemble` object.

		:param normalization:
			If ``True``, the points will be normalized so that concentrate as

		:type normalization:
			bool, default ``True``

		:param title:
			Plot title

		:type title:
			string

		.. caution::

			An initial call to :func:`sample <sample>` is necessary

		.. seealso::

			- :func:`sample <sample>`
			- :func:`hist <hist>`
			- :ref:`full_matrix_models`
			- :ref:`banded_matrix_models`
		"""

		if not self.list_of_samples:
			raise ValueError("list_of_samples is empty, you must sample first")

		fig, ax = plt.subplots(1, 1)
		points = self.list_of_samples[-1].copy()

		if self.name in ("hermite", "laguerre", "jacobi"):

			if self.name == "hermite":

				if normalization:

					if self.mode == "banded":
						N	= self.params['size']

						points = (points - self.params['loc'])/\
											(np.sqrt(0.5)*self.params['scale'])

					elif self.mode == "full":
						N = self.params['N']

					points /= np.sqrt(self.beta * N)

					x = np.linspace(-2,2,100)
					ax.plot(x, semi_circle_law(x),
									'r-', lw=2, alpha=0.6,
									label=r'$f_{sc}$')

			elif self.name == "laguerre":

				if normalization:

					if self.mode == "banded":
						N	= self.params['size']
						M = 2/self.beta * self.params['shape'] + N -1

						points /= 0.5*self.params['scale']

					elif self.mode == "full":
						N = self.params['N']
						M	= self.params['M']

					points /= self.beta * M

					x = np.linspace(1e-3, np.max(points)+0.3, 200)
					ax.plot(x, marcenko_pastur_law(x, M, N),
									'r-', lw=2, alpha=0.6,
									label=r'$f_{MP}$')

			elif self.name == "jacobi":

				if normalization:

					if self.mode == "banded":
						N	= self.params['size']
						M_1 = 2/self.beta * self.params['a'] + N -1
						M_2 = 2/self.beta * self.params['b'] + N -1

					elif self.mode == "full":
						N	= self.params['N']
						M_1 = self.params['M_1']
						M_2 = self.params['M_2']

					eps = 1e-5
					x = np.linspace(eps, 1.0-eps, 500)
					ax.plot(x, wachter_law(x, M_1, M_2, N),
									'r-', lw=2, alpha=0.6,
									label='Wachter Law')

			ax.scatter(points, np.zeros(len(points)), c='blue', label="sample")

		elif self.name in ("circular", "ginibre"):

			if self.name == "circular":

				unit_circle = plt.Circle((0,0), 1, color='r', fill=False)
				ax.add_artist(unit_circle)

				ax.set_xlim([-1.3, 1.3])
				ax.set_ylim([-1.3, 1.3])
				ax.set_aspect('equal')

			if self.name == "ginibre":

				if normalization:

					points /= np.sqrt(self.params['N'])

					unit_circle = plt.Circle((0,0), 1, color='r', fill=False)
					ax.add_artist(unit_circle)

					ax.set_xlim([-1.3, 1.3])
					ax.set_ylim([-1.3, 1.3])
					ax.set_aspect('equal')

			ax.scatter(points.real, points.imag, c='blue', label="sample")

		str_title = "Last realization of {} ensemble with {} points {}".format(
									self.name,
									self.params['N'] if ("N" in self.params) else self.params['size'],
									r"($\beta={}$)".format(self.beta) if self.name!="ginibre" else "")
		plt.title(title if title else str_title)
		ax.legend(loc='best', frameon=False)
		plt.show()


	def hist(self, normalization=True, title=""):
		""" Display the histogram of the last realization of corresponding :class:`BetaEnsemble` object.

		:param normalization:
			If ``True``, the points will be normalized so that concentrate as

		:type normalization:
			bool, default ``True``

		:param title:
			Plot title

		:type title:
			string

		.. caution::

			An initial call to :func:`sample <sample>` is necessary

		.. seealso::

			- :func:`sample <sample>`
			- :func:`plot <plot>`
			- :ref:`full_matrix_models`
			- :ref:`banded_matrix_models`
		"""

		if not self.list_of_samples:
			raise ValueError("list_of_samples is empty, you must sample first")

		fig, ax = plt.subplots(1, 1)
		points = self.list_of_samples[-1].copy()
		# points = np.array(self.list_of_samples).flatten()

		if self.name == "hermite":

			if normalization:

				if self.mode == "banded":
					N	= self.params['size']

					points = (points - self.params['loc'])/\
										(np.sqrt(0.5)*self.params['scale'])

				elif self.mode == "full":
					N = self.params['N']

				points /= np.sqrt(self.beta * N)

				x = np.linspace(-2,2,100)
				ax.plot(x, semi_circle_law(x),
								'r-', lw=2, alpha=0.6,
								label=r'$f_{sc}$')

		elif self.name == "laguerre":

			if normalization:

				if self.mode == "banded":
					N	= self.params['size']
					M = 2/self.beta * self.params['shape'] + N -1

					points /= 0.5*self.params['scale']

				elif self.mode == "full":
					N = self.params['N']
					M	= self.params['M']

				points /= self.beta * M

				x = np.linspace(1e-3,np.max(points)+0.3,100)
				ax.plot(x, marcenko_pastur_law(x, M, N),
								'r-', lw=2, alpha=0.6,
								label=r'$f_{MP}$')

		elif self.name == "jacobi":

			if normalization:

				if self.mode == "banded":
					N	= self.params['size']
					M_1 = 2/self.beta * self.params['a'] + N -1
					M_2 = 2/self.beta * self.params['b'] + N -1

				elif self.mode == "full":
					N	= self.params['N']
					M_1 = self.params['M_1']
					M_2 = self.params['M_2']

				eps = 1e-5
				x = np.linspace(eps,1.0-eps,500)
				ax.plot(x, wachter_law(x, M_1, M_2, N),
								'r-', lw=2, alpha=0.6,
								label='Wachter Law')

		elif self.name == "circular":

			if normalization:

				points = np.angle(points)

				ax.axhline(y=1/(2*np.pi),
									color='r',
									label=r"$\frac{1}{2\pi}$")

		elif self.name == "ginibre":

			raise ValueError("No 'hist' method for Ginibre.")

		ax.hist(points,
						bins=30, density=1,
						facecolor='blue', alpha=0.5,
						label='hist')

		str_title = "Histogram of {} ensemble with {} points {}".format(
									self.name,
										self.params['N'] if ("N" in self.params) else self.params['size'],
										r"($\beta={}$)".format(self.beta) if self.name!="ginibre" else "")
		plt.title(title if title else str_title)

		ax.legend(loc='best', frameon=False)
		plt.show()
		# fig.savefig('foo.pdf')

	# def kernel(self, list_of_points):
	# 	# return the matrix [K(x,y)]_x,y in list_of_points
	# 	# maybe plot the heatmap
	# 	if self.beta != 2:
	# 		raise ValueError("Invalid beta parameter, {} != 2. The OPE is not a DPP, there is no notion of kernel".format(self.beta))
	# 	else:
	# 		pass

	def __check_name_validity(self):

		supported_ensembles = ("hermite, laguerre, jacobi,\
													circular,\
													ginibre")

		if self.name not in supported_ensembles:
			str_list = ["- {}".format(OPE) for OPE in supported_ensembles]
			raise ValueError("\n".join(["Supported OPEs:"] + str_list))

	def __check_beta_validity(self):

		if not (self.beta > 0):
			raise ValueError("beta must be positive")

		elif self.name == "circular":
			if not isinstance(self.beta, int):
				raise ValueError("Invalid beta parameter. For cicurlar ensembles, DPPy only treats positive integers. Given beta={}".format(self.beta))

		elif self.name == "ginibre":
			if not (self.beta == 2):
				raise ValueError("Invalid beta parameter. For the Ginibre ensemble beta must be equal to 2. Given beta={}".format(self.beta))

	def __check_mode_validity(self):

		if self.mode not in ("banded", "full"):
			raise ValueError("Invalid mode attribute. Use 'full'(default) or 'banded'.\nGiven {}".format(self.mode))

	def __check_params_validity(self):

		if self.mode == "banded":

			if self.name == "hermite":
				if ("loc" not in self.params) |\
					 ("scale" not in self.params):
					raise ValueError("Missing 'loc' or 'scale' parameter.\nGiven {}".format(self.params))
				else:
					np.random.normal(self.params["loc"], self.params["scale"])

			elif self.name == "laguerre":
				if ("shape" not in self.params) |\
					 ("scale" not in self.params):
					raise ValueError("Missing 'shape' or 'scale' parameter.\nGiven {}".format(self.params))
				else:
					np.random.gamma(self.params["shape"], scale=self.params["scale"])

			elif self.name == "jacobi":
				if ('a' not in self.params) |\
					 ('b' not in self.params):
					raise ValueError("Missing 'a' or 'b' parameter.\nGiven {}".format(self.params))
				else:
					np.random.beta(self.params["a"], self.params["b"])

			elif self.name == "circular":
				pass

			elif self.name == "ginibre":  
				raise ValueError("Invalid 'mode'. Ginibre has no banded model. Use 'mode=full'.")

		elif self.mode == "full":

			if self.beta not in (1, 2, 4):

				raise ValueError("Invalid match between 'mode' and 'beta'. Sampling using 'mode=full' computes the eigenvalues of a fully filled random matrix and refers to beta = 1, 2 or 4.\nGiven beta={}".format(self.beta))

			else:

				if "N" not in self.params:
						raise ValueError("Missing key 'N' in the dict of sampling parameters. It corresponds to the number of points of the ensemble i.e. the size of the matrix to be diagonalized.\nGiven {}".format(self.params))

				if self.name == "hermite":
					pass

				elif self.name == "laguerre":

					if ('M' not in self.params):
						raise ValueError("Missing key 'M', with M>=N.\nGiven {}".format(self.params))

					elif self.params['M'] < self.params['N']:
						raise ValueError("M<N instead of M>=N.\nGiven {}".format(self.params))

				elif self.name == "jacobi":

					if ('M_1' not in self.params) |\
						 ('M_2' not in self.params):
						raise ValueError("Missing keys 'M_1', 'M_2', with M_1, M_2>=N.\nGiven {}".format(self.params))

					elif (self.params['M_1'] < self.params['N']) |\
							 (self.params['M_2'] < self.params['N']):
						raise ValueError("M_1 or M_2<N instead of M_1 and M_2 >= N.\nGiven {}".format(self.params))

				elif self.name == "circular":

					if "haar_mode" not in self.params:
						raise ValueError("Missing 'haar_mode' parameter under 'mode=full'. Use 'haar_mode=hermite'(default) or 'QR'.\nGiven {}".format(self.params))

					elif self.params["haar_mode"] not in ("hermite", "QR"):
						raise ValueError("Invalid 'haar_mode' parameter when. Use 'hermite' or 'QR'.\nGiven {}".format(self.params))

				elif self.name == "ginibre":
					pass