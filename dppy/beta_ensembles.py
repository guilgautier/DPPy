# coding: utf-8

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

from re import findall as re_findall # to convert class names to string

import dppy.random_matrices as rm

class BetaEnsemble:
	""" :math:`\\beta`-Ensemble object parametrized by

	:param beta:
		:math:`\\beta >= 0` inverse temperature parameter.
		
		The default ``beta=2`` corresponds to the DPP case, see :ref:`beta_ensembles_definition_OPE`
	:type beta:
		int, float, default :math:`2`

	.. seealso::

		- :math:`\\beta`-Ensembles :ref:`definition <beta_ensembles_definition>`
	"""

	def __init__(self, beta=2):

		self.beta = beta
		self.__check_beta_non_negative()

		# Split object name at uppercase
		self.name = ' '.join(re_findall('[A-Z][^A-Z]*', self.__class__.__name__)) 
		self.params = {'size_N':10} # Number of points and reference measure params

		self.sampling_mode = '' # 
		self.list_of_samples = []

	def _str_title(self):
		return r'Realization of {} points of {} with $\beta={}$'.format(									self.params['size_N'], self.name, self.beta)

	def __str__(self):
		str_info = ('{} with beta = {}'.format(self.name, self.beta),
								'sampling parameters = {}'.format(self.params),
								'number of samples = {}'.format(len(self.list_of_samples)))

		return '\n'.join(str_info)

	def __check_beta_non_negative(self):
		if not (self.beta >= 0):
			err_print = ('Invalid `beta` argument:',
									'`beta` must be non negative. Given: {}'.format(self.beta))
			raise ValueError('\n'.join(err_print))
		else:
			pass

	# def info(self):
	# 	""" Print infos about the :class:`BetaEnsemble` object
	# 	"""
	# 	print(self.__str__())

	def flush_samples(self):
		""" Empty the ``list_of_samples`` attribute.
		"""
		self.list_of_samples = []

class HermiteEnsemble(BetaEnsemble):
	""" Hermite Ensemble object

	.. seealso::

		- :ref:`Full matrix model <hermite_ensemble_full>` for Hermite ensemble
		- :ref:`Tridiagonal matrix model <hermite_ensemble_banded>` for Hermite ensemble
	"""

	def __init__(self, beta=2):

		super().__init__(beta=beta)

		# Default parameters for ``loc`` and ``scale`` correspond to the reference measure N(0,2) in the full matrix model
		params = {'loc':0.0, 'scale':np.sqrt(2.0), 'size_N':10}
		self.params.update(params)

	def sample_full_model(self, size_N=10):
		""" Sample from :ref:`tridiagonal matrix model <Hermite_ensemble_full>` for Hermite ensemble. Only available for ``HermiteEnsemble.beta`` attribute :math:`\\beta\\in\\{1, 2, 4\\}` and the degenerate case :math:`\\beta=0` corresponding to i.i.d. points from the Gaussian :math:`\\mathcal{N}(\\mu,\\sigma)` reference measure

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized
		:type size_N:
			int, default :math:`10`

		.. note::

			The reference measure associated with the :ref:`full matrix model <hermite_ensemble_full>` is :math:`\\mathcal{N}(0,2)`.
			For this reason, in the ``sampling_params`` attribute, the values of the parameters are set to ``loc``:math:`=0` and ``scale``:math:`=\\sqrt{2}`.

			To compare :func:`sample_banded_model <sample_banded_model>` with :func:`sample_full_model <sample_full_model>` simply use the ``size_N`` parameter.

		.. seealso::

			- :ref:`Full matrix model <hermite_ensemble_full>` for Hermite ensemble
			- :func:`sample_banded_model <sample_banded_model>`
		"""

		self.sampling_mode = 'full'
		params = {'loc':0.0, 'scale':np.sqrt(2.0), 'size_N':size_N}
		self.params.update(params)

		if self.beta == 0: # Answer issue #28 raised by @rbardenet
			sampl = np.random.normal(loc=params['loc'], scale=params['scale'],
															size=params['size_N'])
		else:
			sampl = rm.hermite_sampler_full(N=params['size_N'], beta=self.beta)

		self.list_of_samples.append(sampl)

	def sample_banded_model(self, loc=0.0, scale=np.sqrt(2.0), size_N=10):
		""" Sample from :ref:`tridiagonal matrix model <Hermite_ensemble_full>` for Hermite Ensemble. Available for ``HermiteEnsemble.beta`` attribute :math:`\\beta>0` and the degenerate case :math:`\\beta=0` corresponding to i.i.d. points from the Gaussian :math:`\\mathcal{N}(\\mu,\\sigma)` reference measure

		:param loc:
			Mean of :math:`\\mu` of the Gamma :math:`\\mathcal{N}(\\mu, \\sigma)`
		:type loc:
			float, default :math:`0`

		:param scale:
			Standard deviation :math:`\\sigma` of the Gamma :math:`\\mathcal{N}(\\mu, \\sigma^2)`
		:type scale:
			float, default :math:`\\sqrt{2}`

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized
		:type size_N:
			int, default :math:`10`

		.. note::

			The reference measure associated with the :ref:`full matrix model <hermite_ensemble_full>` is :math:`\\mathcal{N}(0,2)`.
			For this reason, in the ``sampling_params`` attribute, the default values are set to ``loc``:math:`=0` and ``scale``:math:`=\\sqrt{2}`.

			To compare :func:`sample_banded_model <sample_banded_model>` with :func:`sample_full_model <sample_full_model>` simply use the ``size_N`` parameter.

		.. seealso::

			- :ref:`Tridiagonal matrix model <hermite_ensemble_banded>` for Hermite ensemble
			- :cite:`DuEd02` II-C
			- :func:`sample_full_model <sample_full_model>`
		"""

		self.sampling_mode = 'banded'
		params = {'loc':loc, 'scale':scale, 'size_N':size_N}
		self.params.update(params)

		if self.beta == 0: # Answer issue #28 raised by @rbardenet
			sampl = np.random.normal(loc=params['loc'], scale=params['scale'],
															size=params['size_N'])
		else:
			sampl = rm.mu_ref_normal_sampler_tridiag(loc=params['loc'],
																					scale=params['scale'],
																					beta=self.beta,
																					size=params['size_N'])

		self.list_of_samples.append(sampl)

	def normalize_points(self, points):
		""" Normalize points obtained after sampling to match the limiting distribution i.e. semi-circle

			.. math::

				f(x) = \\frac{1}{2\\pi} \\sqrt{4-x^2}

		:param points:
			A sample from Hermite ensemble, accessible through the ``list_of_samples`` attributes
		:type points:
			array_like

		- If sampled using :func:`sample_banded_model <sample_banded_model>` with reference measure :math:`\\mathcal{N}(\\mu,\\sigma)`

			.. math::

				x \\mapsto \\sqrt{2}\\frac{x-\\mu}{\\sigma}
				\\quad \\text{and} \\quad
				x \\mapsto \\frac{x}{\\beta N}

		- If sampled using :func:`sample_full_model <sample_full_model>`

			.. math::
				
				x \\mapsto \\frac{x}{\\beta N}

		.. note::

			This method is called in :func:`plot <plot>` and :func:`hist <hist>` when ``normalization=True``.
		"""

		if self.sampling_mode == 'banded':
			points -= self.params['loc']
			points /= np.sqrt(0.5)*self.params['scale']

		else: # 'full'
			pass

		points /= np.sqrt(self.beta * self.params['size_N'])

		return points

	def __display_and_normalization(self, display_type, normalization):

		if not self.list_of_samples:
			raise ValueError('Empty attribute `list_of_samples`, sample first!')
		else:
			points = self.list_of_samples[-1].copy() # Pick last sample

		fig, ax = plt.subplots(1, 1)
		# Title
		plt.title(self._str_title())

		if normalization:
			points = self.normalize_points(points)

			# Limiting distribution: semi circle law
			x = np.linspace(-2, 2, 100)
			ax.plot(x, rm.semi_circle_law(x), 
						'r-', lw=2, alpha=0.6,
						label=r'$f_{semi-circle}$')
		else:
			pass

		if display_type == 'scatter':
			ax.scatter(points, np.zeros(points.shape[0]),
								c='blue', 
								label='sample')

		elif display_type == 'hist':
			ax.hist(points, bins=30, density=1,
						facecolor='blue', alpha=0.5,
						label='hist')
		else:
			pass

		plt.legend(loc='best', frameon=False)

	def plot(self, normalization=True):
		""" Display the last realization of the :class:`HermiteEnsemble` object

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: semi-circle using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`normalize_points <normalize_points>`
			- :func:`hist <hist>`
			- :ref:`Full matrix model <hermite_ensemble_full>` for Hermite ensemble
			- :ref:`Tridiagonal matrix model <hermite_ensemble_banded>` for Hermite ensemble
		"""

		self.__display_and_normalization('scatter', normalization)

	def hist(self, normalization=True):
		""" Display the histogram of the last realization of the :class:`HermiteEnsemble` object.

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: semi-circle using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`normalize_points <normalize_points>`
			- :func:`plot <plot>`
			- :ref:`Full matrix model <hermite_ensemble_full>` for Hermite ensemble
			- :ref:`Tridiagonal matrix model <hermite_ensemble_banded>` for Hermite ensemble
		"""
		self.__display_and_normalization('hist', normalization)
		


class LaguerreEnsemble(BetaEnsemble):
	""" Laguerre Ensemble object

	.. seealso::

		- :ref:`Full matrix model <laguerre_ensemble_full>` for Laguerre ensemble
		- :ref:`Tridiagonal matrix model <laguerre_ensemble_banded>` for Laguerre ensemble
	"""

	def __init__(self, beta=2):

		super().__init__(beta=beta)

		params = {'shape':0.0, 'scale':2.0, 'size_N':10, 'size_M':100}
		self.params.update(params)

	def sample_full_model(self, size_N=10, size_M=100):
		""" Sample from :ref:`full matrix model <Laguerre_ensemble_full>` for Laguerre ensemble. Only available for ``LaguerreEnsemble.beta`` attribute :math:`\\beta\\in\\{1, 2, 4\\}` and the degenerate case :math:`\\beta=0` corresponding to i.i.d. points from the :math:`\\Gamma(k,\\theta)` reference measure

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized.
			First dimension of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_ensemble_full>`.
		:type size_N:
			int, default :math:`10`

		:param size_M:
			Second dimension :math:`M` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_ensemble_full>`.

		:type size_M:
			int, default :math:`100`

		.. note::

			The reference measure associated with the :ref:`full matrix model <laguerre_ensemble_full>` is :math:`\\Gamma\\left(\\frac{\\beta}{2}(M-N+1), 2\\right)`.
			For this reason, in the ``LaguerreEnsemble.sampling_params`` attribute, the values of the parameters are set to ``shape``:math:`=\\frac{\\beta}{2}(M-N+1)` and ``scale``:math:`=2`.

			To compare :func:`sample_banded_model <sample_banded_model>` with :func:`sample_full_model <sample_full_model>` simply use the ``size_N`` and ``size_M`` parameters.

		.. seealso::

			- :ref:`Full matrix model <Laguerre_ensemble_full>` for Laguerre ensemble
			- :func:`sample_banded_model <sample_banded_model>`
		"""

		self.sampling_mode = 'full'
		
		if size_M >= size_N:
			shape, scale = 0.5*self.beta*(size_M-size_N+1), 2.0

		else:
			err_print = ('Must have M >= N.',
									'Given: M={} < N={}'.format(size_M, size_N))
			raise ValueError(' '.join(err_print))

		params = {'shape':shape, 'scale':scale, 'size_N':size_N, 'size_M':size_M}
		self.params.update(params)

		if self.beta == 0: # Answer issue #28 raised by @rbardenet
			sampl = np.random.gamma(shape=params['shape'], scale=params['scale'],
														size=params['size_N'])

		else:
			sampl = rm.laguerre_sampler_full(M=params['size_M'], N=params['size_N'],
																			beta=self.beta)

		self.list_of_samples.append(sampl)

	def sample_banded_model(self, shape=1.0, scale=2.0, size_N=10, size_M=None):
		""" Sample from :ref:`tridiagonal matrix model <Laguerre_ensemble_full>` for Laguerre ensemble. Available for ``LaguerreEnsemble.beta`` attribute :math:`\\beta>0` and the degenerate case :math:`\\beta=0` corresponding to i.i.d. points from the :math:`\\Gamma(k,\\theta)` reference measure

		:param shape:
			Shape parameter :math:`k` of the Gamma :math:`\\Gamma(k, \\theta)` reference measure
		:type shape:
			float, default :math:`1`

		:param scale:
			Scale parameter :math:`\\theta` of the Gamma :math:`\\Gamma(k, \\theta)` reference measure
		:type scale:
			float, default :math:`2.0`

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized.
			First dimension :math:`N` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_ensemble_full>`.
		:type size_N:
			int, default :math:`10`

		:param size_M:
			Second dimension :math:`M` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_ensemble_full>`.

		:type size_M:
			int, default None

		- If ``size_M`` is not provided:
			
			In the ``LaguerreEnsemble.sampling_params`` attribute, ``size_M`` is set to 
			``size_M``:math:`= \\frac{2k}{\\beta} + N - 1`, to give an idea of the corresponding second dimension :math:`M`.
			

		- If ``size_M`` is provided:
			
			In the ``LaguerreEnsemble.sampling_params`` attribute, ``shape`` and ``scale`` are set to: 
			``shape``:math:`=\\frac{1}{2} \\beta (M-N+1)` and 
			``scale``:math:`=2`


		.. note::

			The reference measure associated with the :ref:`full matrix model <laguerre_ensemble_full>` is :math:`\\Gamma\\left(\\frac{\\beta}{2}(M-N+1), 2\\right)`. This explains the role of the ``size_M`` parameter.

			To compare :func:`sample_banded_model <sample_banded_model>` with :func:`sample_full_model <sample_full_model>` simply use the ``size_N`` and ``size_M`` parameters

		.. seealso::

			- :ref:`Tridiagonal matrix model <Laguerre_ensemble_banded>` for Laguerre ensemble
			- :cite:`DuEd02` III-B
			- :func:`sample_full_model <sample_full_model>`
		"""

		self.sampling_mode = 'banded'

		if not size_M: # Default setting
			size_M = 2/self.beta * shape + size_N - 1 if self.beta>0 else np.inf

		elif size_M >= size_N:
			shape, scale = 0.5*self.beta*(size_M-size_N+1), 2.0

		else:
			err_print = ('Must have M >= N.',
									'Given: M={} < N={}'.format(size_M, size_N))
			raise ValueError(' '.join(err_print))
			
		params = {'shape':shape, 'scale':scale, 'size_N':size_N, 'size_M':size_M}
		self.params.update(params)

		if self.beta == 0: # Answer issue #28 raised by @rbardenet
			sampl = np.random.gamma(shape=params['shape'], scale=params['scale'],
														size=params['size_N'])
		else:
			sampl = rm.mu_ref_gamma_sampler_tridiag(shape=params['shape'],
																					scale=params['scale'],
																					beta=self.beta,
																					size=params['size_N'])

		self.list_of_samples.append(sampl)

	def normalize_points(self, points):
		""" Normalize points obtained after sampling to match the limiting distribution i.e. Marcenko Pastur law

		.. math::

			\\frac{1}{2\\pi}
			\\frac{\\sqrt{(\\lambda_+-x)(x-\\lambda_-)}}{cx} 
			1_{[\\lambda_-,\\lambda_+]}
			dx

		where	:math:`c = \\frac{M}{N}` and :math:`\\lambda_\\pm = (1\\pm\\sqrt{c})^2`

		:param points:
			A sample from Laguerre ensemble, accessible through the ``list_of_samples`` attributes
		:type points:
			array_like

		- If sampled using :func:`sample_banded_model <sample_banded_model>` with reference measure :math:`\\Gamma(k,\\theta)`

			.. math::

				x \\mapsto \\frac{2x}{\\theta}
				\\quad \\text{and} \\quad
				x \\mapsto \\frac{x}{\\beta M}

		- If sampled using :func:`sample_full_model <sample_full_model>`

			.. math::
				
				x \\mapsto \\frac{x}{\\beta M}

		.. note::

			This method is called in :func:`plot <plot>` and :func:`hist <hist>` when ``normalization=True``.
		"""

		if self.sampling_mode == 'banded':
			points /= 0.5*self.params['scale']
		else: # self.sampling_mode == 'full':
			pass

		if self.beta > 0:
			points /= self.beta * self.params['size_M']

		return points

	def __display_and_normalization(self, display_type, normalization):

		if not self.list_of_samples:
			raise ValueError('Empty attribute `list_of_samples`, sample first!')
		else:
			points = self.list_of_samples[-1].copy() # Pick last sample

		N, M = [self.params.get(key) for key in ('size_N', 'size_M')]

		fig, ax = plt.subplots(1, 1)
		# Title
		str_ratio = r'with ratio $M/N \approx {}$'.format(M/N)
		# Answers Issue #33 raised by @adrienhardy
		title = '\n'.join([self._str_title(), str_ratio])	 
		plt.title(title)

		if normalization:
			points = self.normalize_points(points)

			# Limiting distribution: Marcenko Pastur law
			x = np.linspace(1e-2, np.max(points)+0.3, 100)
			ax.plot(x, rm.marcenko_pastur_law(x, M, N),
						'r-', lw=2, alpha=0.6,
						label=r'$f_{Marcenko-Pastur}$')
		else:
			pass

		if display_type == 'scatter':
			ax.scatter(points, np.zeros(points.shape[0]),
								c='blue', 
								label='sample')

		elif display_type == 'hist':
			ax.hist(points, bins=30, density=1,
						facecolor='blue', alpha=0.5,
						label='hist')
		else:
			pass

		plt.legend(loc='best', frameon=False)

	def plot(self, normalization=True):
		""" Display the last realization of the :class:`LaguerreEnsemble` object

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: Marcenko-Pastur using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`normalize_points <normalize_points>`
			- :func:`hist <hist>`
			- :ref:`Full matrix model <Laguerre_ensemble_full>` for Laguerre ensemble
			- :ref:`Tridiagonal matrix model <Laguerre_ensemble_banded>` for Laguerre ensemble
		"""

		self.__display_and_normalization('scatter', normalization)

	def hist(self, normalization=True):
		""" Display the histogram of the last realization of the :class:`LaguerreEnsemble` object.

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: Marcenko-Pastur using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`normalize_points <normalize_points>`
			- :func:`plot <plot>`
			- :ref:`Full matrix model <Laguerre_ensemble_full>` for Laguerre ensemble
			- :ref:`Tridiagonal matrix model <Laguerre_ensemble_banded>` for Laguerre ensemble
		"""
		self.__display_and_normalization('hist', normalization)
		


class JacobiEnsemble(BetaEnsemble):
	""" Jacobi Ensemble object

	.. seealso::

		- :ref:`Full matrix model <jacobi_ensemble_full>` for Jacobi ensemble
		- :ref:`Tridiagonal matrix model <jacobi_ensemble_banded>` for Jacobi ensemble
	"""

	def __init__(self, beta=2):

		super().__init__(beta=beta)

		params = {'a':1.0, 'b':1.0, 'size_N':10, 'size_M1':20, 'size_M2':20}
		self.params.update(params)

	def sample_full_model(self, size_N=100, size_M1=150, size_M2=200):
		""" Sample from :ref:`full matrix model <Jacobi_ensemble_full>` for Jacobi ensemble. Only available for ``JacobiEnsemble.beta`` attribute :math:`\\beta\\in\\{1, 2, 4\\}` and the degenerate case :math:`\\beta=0` corresponding to i.i.d. points from the :math:`\\operatorname{Beta}(a,b)` reference measure

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized.
			First dimension of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <jacobi_ensemble_full>`.
		:type size_N:
			int, default :math:`100`

		:param size_M1:
			Second dimension :math:`M_1` of the first matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_ensemble_full>`.
		:type size_M1:
			int, default :math:`150`

		:param size_M2:
			Second dimension :math:`M_2` of the second matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_ensemble_full>`.
		:type size_M2:
			int, default :math:`200`

		.. note::

			The reference measure associated with the :ref:`full matrix model <jacobi_ensemble_full>` is 

			.. math::

				\\operatorname{Beta}\\left(\\frac{\\beta}{2}(M_1-N+1), \\frac{\\beta}{2}(M_2-N+1)\\right)

			For this reason, in the ``JacobiEnsemble.sampling_params`` attribute, the values of the parameters are set to ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

			To compare :func:`sample_banded_model <sample_banded_model>` with :func:`sample_full_model <sample_full_model>` simply use the ``size_N``, ``size_M2`` and ``size_M2`` parameters.

		.. seealso::

			- :ref:`Full matrix model <Jacobi_ensemble_full>` for Jacobi ensemble
			- :func:`sample_banded_model <sample_banded_model>`
		"""

		self.sampling_mode = 'full'

		if (size_M1 >= size_N) and (size_M2 >= size_N):
		# all([var >= size_N for var in [size_M1, size_M2]]
			a = 0.5*self.beta*(size_M1 - size_N + 1)
			b = 0.5*self.beta*(size_M2 - size_N + 1)

		else:
			err_print = ('Must have M1, M2 >= N.',
									'Given: M1={}, M2={} and N={}'.format(size_M1, size_M2, size_N))
			raise ValueError(' '.join(err_print))

		params = {'a':a, 'b':b, 'size_N':size_N, 'size_M1':size_M1, 'size_M2':size_M2}
		self.params.update(params)

		if self.beta == 0: # Answer issue #28 raised by @rbardenet
			sampl = np.random.beta(a=params['a'], b=params['b'],
														size=params['size_N'])

		else:
			sampl = rm.jacobi_sampler_full(M_1=params['size_M1'],
																	M_2=params['size_M2'],
																	N=params['size_N'],
																	beta=self.beta)

		self.list_of_samples.append(sampl)

	def sample_banded_model(self, a=1.0, b=2.0, size_N=10, size_M1=None, size_M2=None):
		""" Sample from :ref:`tridiagonal matrix model <Jacobi_ensemble_full>` for Jacobi ensemble. Available for ``JacobiEnsemble.beta`` attribute :math:`\\beta>0` and the degenerate case :math:`\\beta=0` corresponding to i.i.d. points from the :math:`\\operatorname{Beta}(a,b)` reference measure

		:param shape:
			Shape parameter :math:`k` of the Gamma :math:`\\Gamma(k, \\theta)` reference measure
		:type shape:
			float, default :math:`1`

		:param scale:
			Scale parameter :math:`\\theta` of the Gamma :math:`\\Gamma(k, \\theta)` reference measure
		:type scale:
			float, default :math:`2.0`

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized.
			First dimension :math:`N` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <jacobi_ensemble_full>`.
		:type size_N:
			int, default :math:`10`

		:param size_M1:
			Second dimension :math:`M_1` of the first matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_ensemble_full>`.
		:type size_M1:
			int, default :math:`150`

		:param size_M2:
			Second dimension :math:`M_2` of the second matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_ensemble_full>`.
		:type size_M2:
			int, default :math:`200`

		.. note::

			The reference measure associated with the :ref:`full matrix model <jacobi_ensemble_full>` is :

			.. math::

				\\operatorname{Beta}\\left(\\frac{\\beta}{2}(M_1-N+1), \\frac{\\beta}{2}(M_2-N+1)\\right)

			For this reason, in the ``JacobiEnsemble.sampling_params`` attribute, the values of the parameters are set to ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

			To compare :func:`sample_banded_model <sample_banded_model>` with :func:`sample_full_model <sample_full_model>` simply use the ``size_N``, ``size_M2`` and ``size_M2`` parameters.

		- If ``size_M1`` and ``size_M2`` are not provided:
			
			In the ``JacobiEnsemble.sampling_params`` attribute, ``size_M1,2`` are set to 
			``size_M1``:math:`= \\frac{2a}{\\beta} + N - 1` and ``size_M2``:math:`= \\frac{2b}{\\beta} + N - 1`, to give an idea of the corresponding second dimensions :math:`M_{1,2}`.
			

		- If ``size_M1`` and ``size_M2`` are provided:
			
			In the ``JacobiEnsemble.sampling_params`` attribute, ``a`` and ``b`` are set to: 
			``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and 
			``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

		.. seealso::

			- :ref:`Tridiagonal matrix model <Jacobi_ensemble_banded>` for Jacobi ensemble
			- :cite:`KiNe04` Theorem 2
			- :func:`sample_full_model <sample_full_model>`
		"""

		self.sampling_mode = 'banded'

		if not (size_M1 and size_M2): # default setting

			if self.beta > 0:
				size_M1 = 2/self.beta * a + size_N -1
				size_M2 = 2/self.beta * b + size_N -1

			else:
				size_M1, size_M2 = inf, inf

		elif (size_M1 >= size_N) and (size_M2 >= size_N):
			# all([var >= size_N for var in [size_M1, size_M2]]
			a = 0.5*self.beta*(size_M1 - size_N + 1)
			b = 0.5*self.beta*(size_M2 - size_N + 1)

		else:
			err_print = ('Must have M1, M2 >= N.',
									'Given: M1={}, M2={} and N={}'.format(size_M1, size_M2, size_N))
			raise ValueError(' '.join(err_print))


		params = {'a':a, 'b':b, 'size_N':size_N, 'size_M1':size_M1, 'size_M2':size_M2}
		self.params.update(params)

		if self.beta == 0: # Answer issue #28 raised by @rbardenet
				sampl = np.random.beta(a=params['a'], b=params['b'],
															size=params['size_N'])
		else:
			sampl = rm.mu_ref_beta_sampler_tridiag(a=params['a'],
																					b=params['b'],
																					beta=self.beta,
																					size=params['size_N'])

		self.list_of_samples.append(sampl)

	def __display_and_normalization(self, display_type, normalization):

		if not self.list_of_samples:
			raise ValueError('Empty attribute `list_of_samples`, sample first!')
		else:
			points = self.list_of_samples[-1].copy() # Pick last sample

		N, M_1, M_2 = [self.params.get(key) for key in ('size_N', 'size_M1', 'size_M2')]

		fig, ax = plt.subplots(1, 1)
		# Title
		str_ratios = r'with ratios $M_1/N \approx {:.3f}, M_2/N \approx {:.3f}$'.format(M_1/N, M_2/N) # Answers Issue #33 raised by @adrienhardy
		title = '\n'.join([self._str_title(), str_ratios])
		plt.title(title)

		if normalization:
			# Limiting distribution: Marcenko Pastur law
			eps = 5e-3
			x = np.linspace(eps, 1.0-eps, 500)
			ax.plot(x, rm.wachter_law(x, M_1, M_2, N),
							'r-', lw=2, alpha=0.6,
							label=r'$f_{Wachter}$')
		else:
			pass

		if display_type == 'scatter':
			ax.scatter(points, np.zeros(points.shape[0]),
								c='blue', 
								label='sample')

		elif display_type == 'hist':
			ax.hist(points, bins=30, density=1,
						facecolor='blue', alpha=0.5,
						label='hist')
		else:
			pass

		plt.legend(loc='best', frameon=False)

	def plot(self, normalization=True):
		""" Display the last realization of the :class:`JacobiEnsemble` object

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: Marcenko-Pastur using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`hist <hist>`
			- :ref:`Full matrix model <Jacobi_ensemble_full>` for Jacobi ensemble
			- :ref:`Tridiagonal matrix model <Jacobi_ensemble_banded>` for Jacobi ensemble
		"""

		self.__display_and_normalization('scatter', normalization)

	def hist(self, normalization=True):
		""" Display the histogram of the last realization of the :class:`JacobiEnsemble` object.

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: Marcenko-Pastur using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`normalize_points <normalize_points>`
			- :func:`plot <plot>`
			- :ref:`Full matrix model <Jacobi_ensemble_full>` for Jacobi ensemble
			- :ref:`Tridiagonal matrix model <Jacobi_ensemble_banded>` for Jacobi ensemble
		"""
		self.__display_and_normalization('hist', normalization)



class CircularEnsemble(BetaEnsemble):
	""" Circular Ensemble object

	.. seealso::

		- :ref:`Full matrix model <circular_ensemble_full>` for Circular ensemble
		- :ref:`Quindiagonal matrix model <circular_ensemble_banded>` for Circular ensemble
	"""

	def __init__(self, beta=2):

		super().__init__(beta=beta)
		# Check positive integer!

		params = {'size_N':10}
		self.params.update(params)

	def sample_full_model(self, size_N=10, haar_mode='Hermite'):
		""" Sample from :ref:`tridiagonal matrix model <Circular_ensemble_full>` for Circular ensemble. Only available for ``CircularEnsemble.beta`` attribute :math:`\\beta\\in\\{1, 2, 4\\}` and the degenerate case :math:`\\beta=0` corresponding to i.i.d. uniform points on the unit circle

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized
		:type size_N:
			int, default :math:`10`

		:param haar_mode:
			Sample Haar measure i.e. uniformly on the orthogonal/unitary/symplectic group using:
			- 'QR', 
			- 'Hermite'
		:type haar_mode:
			str, default 'hermite'

		.. seealso::

			- :ref:`Full matrix model <circular_ensemble_full>` for Circular ensemble
			- :func:`sample_banded_model <sample_banded_model>`
		"""

		self.sampling_mode = 'full'
		params = {'size_N':size_N, 'haar_mode':haar_mode}
		self.params.update(params)

		if self.beta == 0: # i.i.d. points uniformly on the circle
			# Answer issue #28 raised by @rbardenet
			sampl = np.exp(2*1j*np.pi*np.random.rand(params['size_N']))
		else:
			sampl = rm.circular_sampler_full(N=params['size_N'],
																			beta=self.beta,
																			haar_mode=params['haar_mode'])

		self.list_of_samples.append(sampl)

	def sample_banded_model(self, size_N=10):
		""" Sample from :ref:`tridiagonal matrix model <Circular_ensemble_full>` for Circular Ensemble. 
		Available for ``CircularEnsemble.beta`` attribute :math:`\\beta\\in\\mathbb{N}^*`, and the degenerate case :math:`\\beta=0` corresponding to i.i.d. uniform points on the unit circle

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized
		:type size_N:
			int, default :math:`10`

		.. note::

			To compare :func:`sample_banded_model <sample_banded_model>` with :func:`sample_full_model <sample_full_model>` simply use the ``size_N`` parameter.

		.. seealso::

			- :ref:`Quindiagonal matrix model <circular_ensemble_banded>` for Circular ensemble
			- :func:`sample_full_model <sample_full_model>`
		"""

		self.sampling_mode = 'banded'
		params = {'size_N':size_N}
		self.params.update(params)

		if self.beta == 0: # i.i.d. points uniformly on the circle
			# Answer issue #28 raised by @rbardenet
			sampl = np.exp(2*1j*np.pi*np.random.rand(params['size_N']))
		else:
			sampl = rm.mu_ref_unif_unit_circle_sampler_quindiag(beta=self.beta,
																		  									size=params['size_N'])

		self.list_of_samples.append(sampl)

	def __display_and_normalization(self, display_type):

		if not self.list_of_samples:
			raise ValueError('Empty attribute `list_of_samples`, sample first!')
		else:
			points = self.list_of_samples[-1].copy() # Pick last sample

		fig, ax = plt.subplots(1, 1)

		# Title
		samp_mod = ''
		if self.beta ==0:
			samp_mod = r'i.i.d samples from $\mathcal{U}_{[0,2\pi]}$'
		elif self.sampling_mode == 'full':
			samp_mod = 'full matrix model with haar_mode={}'.format(self.params['haar_mode'])
		else: # self.sampling_mode == 'banded':
			samp_mod = 'quindiag model'

		title = '\n'.join([self._str_title(), 'using {}'.format(samp_mod)])
		plt.title(title)

		if display_type == 'scatter':
			# Draw unit circle
			unit_circle = plt.Circle((0,0), 1, color='r', fill=False)
			ax.add_artist(unit_circle)

			ax.set_xlim([-1.3, 1.3])
			ax.set_ylim([-1.3, 1.3])
			ax.set_aspect('equal')
			ax.scatter(points.real, points.imag, c='blue', label='sample')

		elif display_type == 'hist':
			points = np.angle(points)

			# Uniform distribution on [0, 2pi]
			ax.axhline(y=1/(2*np.pi), color='r', label=r'$\frac{1}{2\pi}$')

			ax.hist(points, bins=30, density=1,
						facecolor='blue', alpha=0.5,
						label='hist')
		else:
			pass

		plt.legend(loc='best', frameon=False)

	def plot(self):
		""" Display the last realization of the :class:`CircularEnsemble` object

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`hist <hist>`
			- :ref:`Full matrix model <circular_ensemble_full>` for Circular ensemble
			- :ref:`Quindiagonal matrix model <circular_ensemble_banded>` for Circular ensemble
		"""

		self.__display_and_normalization('scatter')

	def hist(self):
		""" Display the histogram of the last realization of the :class:`CircularEnsemble` object.

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: semi-circle using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`, :func:`sample_banded_model <sample_banded_model>`
			- :func:`plot <plot>`
			- :ref:`Full matrix model <circular_ensemble_full>` for Circular ensemble
			- :ref:`Quindiagonal matrix model <circular_ensemble_banded>` for Circular ensemble
		"""
		self.__display_and_normalization('hist')
		

class GinibreEnsemble(BetaEnsemble):
	""" Ginibre Ensemble object

	.. seealso::

		- :ref:`Full matrix model <ginibre_ensemble_full>` for Ginibre ensemble
	"""

	def __init__(self, beta=2):

		super().__init__(beta=beta)
		# Check beta=2!

		params = {'size_N':10}
		self.params.update(params)

	def sample_full_model(self, size_N=10):
		""" Sample from :ref:`full matrix model <Ginibre_ensemble_full>` for Ginibre ensemble. Only available for ``GinibreEnsemble.beta`` attribute :math:`\\beta=2`

		:param size_N:
			Number :math:`N` of points i.e. size of the matrix to be diagonalized
		:type size_N:
			int, default :math:`10`

		.. seealso::

			- :ref:`Full matrix model <ginibre_ensemble_full>` for Ginibre ensemble
			- :func:`sample_banded_model <sample_banded_model>`
		"""

		self.params.update({'size_N':size_N})
		sampl = rm.ginibre_sampler_full(N=self.params['size_N'])

		self.list_of_samples.append(sampl)

	def plot(self, normalization=True):
		""" Display the last realization of the :class:`GinibreEnsemble` object

		:param normalization:

			If ``True``, the points are normalized so as to concentrate as the lititing distribution: semi-circle using :func:`normalize_points <normalize_points>`

		:type normalization:
			bool, default ``True``

		.. seealso::

			- :func:`sample_full_model <sample_full_model>`
			- :ref:`Full matrix model <ginibre_ensemble_full>` for Ginibre ensemble ensemble
		"""	

		if not self.list_of_samples:
			raise ValueError('Empty attribute `list_of_samples`, sample first!')
		else:
			points = self.list_of_samples[-1].copy() # Pick last sample

		# Plot
		fig, ax = plt.subplots(1, 1)
	
		plt.title(self._str_title()) # Title

		if normalization:
			points /= np.sqrt(self.params['size_N'])
			# Draw unit circle
			unit_circle = plt.Circle((0,0), 1, color='r', fill=False)
			ax.add_artist(unit_circle)

			ax.set_xlim([-1.3, 1.3])
			ax.set_ylim([-1.3, 1.3])
			ax.set_aspect('equal')

		ax.scatter(points.real, points.imag, c='blue', label='sample')

		plt.legend(loc='best', frameon=False)