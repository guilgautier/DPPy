# coding: utf-8
try: # Local import
	from .random_matrices import *
except (SystemError, ImportError):
	from random_matrices import *
import matplotlib.pyplot as plt

class BetaEnsemble:
	""" Discrete DPP object parametrized by

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

		self.sampling_mode = None
		self.sampling_params = {}
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
																			self.sampling_mode,
																			self.sampling_params,
																			len(self.list_of_samples))

	def info(self):
		""" Print infos about the :class:`BetaEnsemble` object
		"""
		print(self.__str__())

	def flush_samples(self):
		""" Empty the ``BetaEnsemble.list_of_samples`` attribute.
		"""
		self.list_of_samples = []

	def sample(self, sampling_mode='full', **sampling_params):
		""" Sample exactly from the corresponding :class:`BetaEnsemble <BetaEnsemble>` object by computing the eigenvalues of random matrices.

		:param sampling_params:

			- ``sampling_mode='full'``:
			- ``sampling_mode='banded'``:

		:type sampling_params:
			string, default ``'full'``

		:param sampling_params:
			Dictionary containing the parametrization of the underlying :class:`BetaEnsemble <BetaEnsemble>` object viewed as the eigenvalues of a full or banded random matrix.

			For ``sampling_mode='full'``, the ``'N'`` key refers to the number of points i.e. the size of the matrix to be diagonalized.

				- for ``BetaEnsemble.name='hermite'``

					``sampling_params={'N':N}``

				- for ``BetaEnsemble.name='laguerre'``

					``sampling_params={'M':M, 'N':N}`` where :math:`M \geq N`

				- for ``BetaEnsemble.name='jacobi'``

					``sampling_params={'M_1': M_1, 'M_2':M_2, 'N':N}`` where :math:`M_{1,2}\geq N`

				- for ``BetaEnsemble.name='circular'``

					``sampling_params={'N':N, 'mode':'QR'/'hermite'}``

				- for ``BetaEnsemble.name='ginibre'``

					``sampling_params={'N':N}``

			For ``sampling_mode='banded'``, the ``'size'`` key refers to the number of points i.e. the size of the matrix to be diagonalized.

				- for ``BetaEnsemble.name='hermite'``

					- ``sampling_params={'loc':, 'scale':, 'size':}``, where ``'loc', 'scale'`` are respectively the mean and standard deviation of the corresponding Gaussian reference measure. To recover the full matrix model take, ``loc`` :math:`=0`, ``scale`` :math:`=\sqrt{2}` and ``size``:math:`=N`.

				- for ``BetaEnsemble.name='laguerre'``

					- ``sampling_params={'shape':, 'scale':, 'size':}``, where ``'shape', 'scale'`` are respectively the shape and standard deviation of the corresponding Gamma reference measure. To recover the full matrix model take ``shape`` :math:`=\\frac{1}{2} \\beta (M-N+1)`, ``scale``:math:`=2` and ``size`` :math:`=N`.

				- for ``BetaEnsemble.name='jacobi'``

					- ``sampling_params={'a':, 'b':, 'size':}``, where ``'a', 'b'`` are the respective parameters of the corresponding Beta reference measure. To recover the full matrix model take :math:`a=\\frac{1}{2} \\beta (M_1-N+1)`, :math:`b=\\frac{1}{2} \\beta (M_2-N+1)` and ``size``:math:`=N`.

				- for ``BetaEnsemble.name='circular'``

					- ``sampling_params={size:}``.

		:type sampling_params:
			dict

		.. seealso::

			- :func:`flush_samples <flush_samples>`
			- :ref:`full_matrix_models` and :ref:`banded_matrix_models`
		"""

		self.sampling_mode = sampling_mode
		self.__check_sampling_mode_validity()

		self.sampling_params = sampling_params
		self.__check_params_validity()

		if self.sampling_mode == "banded":

			if self.name == "hermite":
					sampl = muref_normal_sampler_tridiag(loc=self.sampling_params["loc"],
																					scale=self.sampling_params["scale"],
																					beta=self.beta,
																					size=self.sampling_params["size"])

			elif self.name == "laguerre":
					sampl = mu_ref_gamma_sampler_tridiag(
																					shape=self.sampling_params["shape"],
																					scale=self.sampling_params["scale"],
																					beta=self.beta,
																					size=self.sampling_params["size"])

			elif self.name == "jacobi":
					sampl = mu_ref_beta_sampler_tridiag(a=self.sampling_params["a"],
																						b=self.sampling_params["b"],
																						beta=self.beta,
																						size=self.sampling_params["size"])

			elif self.name == "circular":
					sampl = mu_ref_unif_unit_circle_sampler_quindiag(beta=self.beta,
																		  			size=self.sampling_params["size"])

			elif self.name == "ginibre":

					raise ValueError("In valid 'sampling_mode' argument. No banded model for Ginibre ensemble. Use 'full'.\nGiven {}".format(sampling_mode))

		elif self.sampling_mode == "full":

			if self.name == "hermite":
				sampl = hermite_sampler_full(N=self.sampling_params['N'],
																		beta=self.beta)

			elif self.name == "laguerre":
				sampl = laguerre_sampler_full(M=self.sampling_params['M'],
																			N=self.sampling_params['N'],
																			beta=self.beta)

			elif self.name == "jacobi":
				sampl = jacobi_sampler_full(M_1=self.sampling_params['M_1'],
																		M_2=self.sampling_params['M_2'],
																		N=self.sampling_params['N'],
																		beta=self.beta)

			elif self.name == "circular":
				sampl = circular_sampler_full(N=self.sampling_params['N'],
																			beta=self.beta,
																			mode=self.sampling_params["mode"])

			elif self.name == "ginibre":
				sampl = ginibre_sampler_full(N=self.sampling_params['N'])

		self.list_of_samples.append(sampl)


	def plot(self, normalization=True):
		""" Display the histogram of the corresponding :class:`BetaEnsemble` object

		:param normalization:
			If ``True``, the points will be normalized so that concentrate as

		:type normalization:
			bool, default ``True``

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

					if self.sampling_mode == "banded":
						N	= self.sampling_params['size']

						points = (points - self.sampling_params['loc'])/\
											(np.sqrt(0.5)*self.sampling_params['scale'])

					elif self.sampling_mode == "full":
						N = self.sampling_params['N']

					points /= np.sqrt(self.beta * N)

					x = np.linspace(-2,2,100)
					ax.plot(x, semi_circle_law(x),
									'r-', lw=2, alpha=0.6,
									label=r'$f_{sc}$')

			elif self.name == "laguerre":

				if normalization:

					if self.sampling_mode == "banded":
						N	= self.sampling_params['size']
						M = 2/self.beta * self.sampling_params['shape'] + N -1

						points /= 0.5*self.sampling_params['scale']

					elif self.sampling_mode == "full":
						N = self.sampling_params['N']
						M	= self.sampling_params['M']

					points /= self.beta * M

					x = np.linspace(1e-3, np.max(points)+0.3, 200)
					ax.plot(x, marcenko_pastur_law(x, M, N),
									'r-', lw=2, alpha=0.6,
									label=r'$f_{MP}$')

			elif self.name == "jacobi":

				if normalization:

					if self.sampling_mode == "banded":
						N	= self.sampling_params['size']
						M_1 = 2/self.beta * self.sampling_params['a'] + N -1
						M_2 = 2/self.beta * self.sampling_params['b'] + N -1

					elif self.sampling_mode == "full":
						N	= self.sampling_params['N']
						M_1 = self.sampling_params['M_1']
						M_2 = self.sampling_params['M_2']

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

					points /= np.sqrt(self.sampling_params['N'])

					unit_circle = plt.Circle((0,0), 1, color='r', fill=False)
					ax.add_artist(unit_circle)

					ax.set_xlim([-1.3, 1.3])
					ax.set_ylim([-1.3, 1.3])
					ax.set_aspect('equal')

			ax.scatter(points.real, points.imag, c='blue', label="sample")

		str_title = "Last realization of {} ensemble with {} points {}".format(
									self.name,
									self.sampling_params['N'] if ("N" in self.sampling_params) else self.sampling_params['size'],
									r"($\beta={}$)".format(self.beta) if self.name!="ginibre" else "")
		plt.title(str_title)
		ax.legend(loc='best', frameon=False)
		plt.show()


	def hist(self, normalization=True):
		""" Display the histogram of the corresponding :class:`BetaEnsemble` object

		:param normalization:
			If ``True``, the points will be normalized so that concentrate as

		:type normalization:
			bool, default ``True``

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

				if self.sampling_mode == "banded":
					N	= self.sampling_params['size']

					points = (points - self.sampling_params['loc'])/\
										(np.sqrt(0.5)*self.sampling_params['scale'])

				elif self.sampling_mode == "full":
					N = self.sampling_params['N']

				points /= np.sqrt(self.beta * N)

				x = np.linspace(-2,2,100)
				ax.plot(x, semi_circle_law(x),
								'r-', lw=2, alpha=0.6,
								label=r'$f_{sc}$')

		elif self.name == "laguerre":

			if normalization:

				if self.sampling_mode == "banded":
					N	= self.sampling_params['size']
					M = 2/self.beta * self.sampling_params['shape'] + N -1

					points /= 0.5*self.sampling_params['scale']

				elif self.sampling_mode == "full":
					N = self.sampling_params['N']
					M	= self.sampling_params['M']

				points /= self.beta * M

				x = np.linspace(1e-3,np.max(points)+0.3,100)
				ax.plot(x, marcenko_pastur_law(x, M, N),
								'r-', lw=2, alpha=0.6,
								label=r'$f_{MP}$')

		elif self.name == "jacobi":

			if normalization:

				if self.sampling_mode == "banded":
					N	= self.sampling_params['size']
					M_1 = 2/self.beta * self.sampling_params['a'] + N -1
					M_2 = 2/self.beta * self.sampling_params['b'] + N -1

				elif self.sampling_mode == "full":
					N	= self.sampling_params['N']
					M_1 = self.sampling_params['M_1']
					M_2 = self.sampling_params['M_2']

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
										self.sampling_params['N'] if ("N" in self.sampling_params) else self.sampling_params['size'],
										r"($\beta={}$)".format(self.beta) if self.name!="ginibre" else "")
		plt.title(str_title)

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

	def __check_sampling_mode_validity(self):

		if self.sampling_mode not in ("banded", "full"):
			raise ValueError("Invalid sampling_mode attribute. Use 'full'(default) or 'banded'.\nGiven {}".format(self.sampling_mode))

	def __check_params_validity(self):

		if self.sampling_mode == "banded":

			if self.name == "hermite":
				if ("loc" not in self.sampling_params) |\
					 ("scale" not in self.sampling_params):
					raise ValueError("Missing 'loc' or 'scale' parameter.\nGiven {}".format(self.sampling_params))
				else:
					np.random.normal(self.sampling_params["loc"], self.sampling_params["scale"])

			elif self.name == "laguerre":
				if ("shape" not in self.sampling_params) |\
					 ("scale" not in self.sampling_params):
					raise ValueError("Missing 'shape' or 'scale' parameter.\nGiven {}".format(self.sampling_params))
				else:
					np.random.gamma(self.sampling_params["shape"], scale=self.sampling_params["scale"])

			elif self.name == "jacobi":
				if ('a' not in self.sampling_params) |\
					 ('b' not in self.sampling_params):
					raise ValueError("Missing 'a' or 'b' parameter.\nGiven {}".format(self.sampling_params))
				else:
					np.random.beta(self.sampling_params["a"], self.sampling_params["b"])

			elif self.name == "circular":
				pass

			elif self.name == "ginibre":
				raise ValueError("Invalid 'sampling_mode'. Ginibre has no banded model. Use 'sampling_mode=full'.")

		elif self.sampling_mode == "full":

			if self.beta not in (1, 2, 4):

				raise ValueError("Invalid match between 'sampling_mode' and 'beta'. Sampling using 'sampling_mode=full' computes the eigenvalues of a fully filled random matrix and refers to beta = 1, 2 or 4.\nGiven beta={}".format(self.beta))

			else:

				if "N" not in self.sampling_params:
						raise ValueError("Missing key 'N' in the dict of sampling parameters. It corresponds to the number of points of the ensemble i.e. the size of the matrix to be diagonalized.\nGiven {}".format(self.sampling_params))

				if self.name == "hermite":
					pass

				elif self.name == "laguerre":

					if ('M' not in self.sampling_params):
						raise ValueError("Missing key 'M', with M>=N.\nGiven {}".format(self.sampling_params))

					elif self.sampling_params['M'] < self.sampling_params['N']:
						raise ValueError("M<N instead of M>=N.\nGiven {}".format(self.sampling_params))

				elif self.name == "jacobi":

					if ('M_1' not in self.sampling_params) |\
						 ('M_2' not in self.sampling_params):
						raise ValueError("Missing keys 'M_1', 'M_2', with M_1, M_2>=N.\nGiven {}".format(self.sampling_params))

					elif (self.sampling_params['M_1'] < self.sampling_params['N']) |\
							 (self.sampling_params['M_2'] < self.sampling_params['N']):
						raise ValueError("M_1 or M_2<N instead of M_1 and M_2 >= N.\nGiven {}".format(self.sampling_params))

				elif self.name == "circular":

					if "mode" not in self.sampling_params:
						raise ValueError("Missing 'mode' parameter under 'sampling_mode=full'. Use 'mode=hermite'(default) or 'QR'.\nGiven {}".format(self.sampling_params))

					elif self.sampling_params["mode"] not in ("hermite", "QR"):
						raise ValueError("Invalid 'mode' parameter when. Use 'hermite' or 'QR'.\nGiven {}".format(self.sampling_params))

				elif self.name == "ginibre":
					pass