# coding: utf8
""" Implementation of the meta-class :class:`BetaEnsemble` see :math:`\\beta-`:ref:`Ensembles<beta_ensembles>` with children:

- :class:`HermiteEnsemble`
- :class:`LaguerreEnsemble`
- :class:`JacobiEnsemble`
- :class:`CircularEnsemble`
- :class:`GinibreEnsemble`

Such objects have 4 main methods:

- :py:func:`sample_full_model`
- :py:func:`sample_banded_model`
- :py:func:`plot` to display a scatter plot of the last sample and eventually the limiting distribution (after normalization)
- :py:func:`hist` to display a histogram of the last sample and eventually the limiting distribution (after normalization)

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/continuous_dpps/beta_ensembles.html>`_
"""

import abc

from sys import platform as _platform
# https://stackoverflow.com/questions/1854/python-what-os-am-i-running-on
if _platform.startswith('linux'):
    # linux
    pass
elif _platform == "darwin":
    # MAC OS X
    # https://markhneedham.com/blog/2018/05/04/python-runtime-error-osx-matplotlib-not-installed-as-framework-mac/
    # import matplotlib
    # matplotlib.use('TkAgg')
    pass

import matplotlib.pyplot as plt

import numpy as np

from scipy.stats import norm as sp_gaussian
from scipy.stats import gamma as sp_gamma
from scipy.stats import beta as sp_beta

from re import findall as re_findall  # to convert class names to string

import dppy.random_matrices as rm

from dppy.utils import check_random_state


class BetaEnsemble(metaclass=abc.ABCMeta):
    """ :math:`\\beta`-Ensemble object parametrized by

    :param beta:
        :math:`\\beta >= 0` inverse temperature parameter.

        The default :py:attr:`beta`:math:`=2` corresponds to the DPP case,
        see :ref:`beta_ensembles_definition_OPE`
    :type beta:
        int, float, default :math:`2`

    .. seealso::

        - :math:`\\beta`-Ensembles :ref:`definition <beta_ensembles_definition>`
    """

    def __init__(self, beta=2):

        if not (beta >= 0):
            raise ValueError('`beta` must be >=0. Given: {}'.format(self.beta))
        self.beta = beta

        # Split object name at uppercase
        self.name = ' '.join(re_findall('[A-Z][^A-Z]*',
                                        self.__class__.__name__))
        self.params = {'size_N': 10}  # Number of points and ref measure params

        self.sampling_mode = ''
        self.list_of_samples = []

    @property
    def _str_title(self):
        return r'Realization of {} points of {} with $\beta={}$'.format(
            self.params['size_N'], self.name, self.beta)

    def __str__(self):
        str_info = ('{} with beta = {}'.format(self.name, self.beta),
                    'sampling parameters = {}'.format(self.params),
                    'number of samples = {}'.format(len(self.list_of_samples)))

        return '\n'.join(str_info)

    def flush_samples(self):
        """ Empty the :py:attr:`list_of_samples` attribute.
        """
        self.list_of_samples = []

    @abc.abstractmethod
    def sample_full_model(self):
        """Sample from underlying :math:`\\beta`-Ensemble using the corresponding full matrix model.
        Arguments are the associated matrix dimensions
        """

    @abc.abstractmethod
    def sample_banded_model(self):
        """Sample from underlying :math:`\\beta`-Ensemble using the corresponding banded matrix model.
        Arguments are the associated reference measure's parameters, or the matrix dimensions used in :py:meth:`sample_full_model`
        """

    @abc.abstractmethod
    def plot(self):
        """Display last realization of the underlying :math:`\\beta`-Ensemble.
        For some :math:`\\beta`-Ensembles, a normalization argument is available to display the limiting (or equilibrium) distribution and scale the points accordingly.
        """

    @abc.abstractmethod
    def hist(self):
        """Display histogram of the last realization of the underlying :math:`\\beta`-Ensemble.
        For some :math:`\\beta`-Ensembles, a normalization argument is available to display the limiting (or equilibrium) distribution and scale the points accordingly.
        """


class HermiteEnsemble(BetaEnsemble):
    """ Hermite Ensemble object

    .. seealso::

        - :ref:`Full matrix model <hermite_full_matrix_model>` for Hermite ensemble
        - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` for Hermite ensemble
    """

    def __init__(self, beta=2):

        super().__init__(beta=beta)

        # Default parameters for ``loc`` and ``scale`` correspond to the
        # reference measure N(0,2) in the full matrix model
        params = {'loc': 0.0, 'scale': np.sqrt(2.0),
                  'size_N': 10}
        self.params.update(params)

    def sample_full_model(self, size_N=10,
                          random_state=None):
        """ Sample from :ref:`tridiagonal matrix model <Hermite_full_matrix_model>` for Hermite ensemble.
        Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}`
        and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the Gaussian :math:`\\mathcal{N}(\\mu,\\sigma^2)` reference measure

        :param size_N:
            Number :math:`N` of points = size of the matrix to be diagonalized
        :type size_N:
            int, default :math:`10`

        .. note::

            The reference measure associated with the :ref:`full matrix model <hermite_full_matrix_model>` is :math:`\\mathcal{N}(0,2)`.
            For this reason, in the :py:attr:`sampling_params` attribute, the values of the parameters are set to ``loc``:math:`=0` and ``scale``:math:`=\\sqrt{2}`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` parameter.

        .. seealso::

            - :ref:`Full matrix model <hermite_full_matrix_model>` for Hermite ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'full'
        params = {'loc': 0.0, 'scale': np.sqrt(2.0),
                  'size_N': size_N}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample N i.i.d. gaussian N(0,2)
            sampl = rng.normal(loc=params['loc'],
                               scale=params['scale'],
                               size=params['size_N'])
        else:  # if beta > 0
            sampl = rm.hermite_sampler_full(N=params['size_N'],
                                            beta=self.beta,
                                            random_state=rng)

        self.list_of_samples.append(sampl)

    def sample_banded_model(self, loc=0.0, scale=np.sqrt(2.0), size_N=10,
                            random_state=None):
        """ Sample from :ref:`tridiagonal matrix model <Hermite_full_matrix_model>` for Hermite Ensemble.
        Available for :py:attr:`beta` attribute  :math:`\\beta>0` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the Gaussian :math:`\\mathcal{N}(\\mu,\\sigma^2)` reference measure

        :param loc:
            Mean :math:`\\mu` of the Gaussian :math:`\\mathcal{N}(\\mu, \\sigma^2)`
        :type loc:
            float, default :math:`0`

        :param scale:
            Standard deviation :math:`\\sigma` of the Gaussian :math:`\\mathcal{N}(\\mu, \\sigma^2)`
        :type scale:
            float, default :math:`\\sqrt{2}`

        :param size_N:
            Number :math:`N` of points i.e. size of the matrix to be diagonalized
        :type size_N:
            int, default :math:`10`

        .. note::

            The reference measure associated with the :ref:`full matrix model <hermite_full_matrix_model>` is :math:`\\mathcal{N}(0,2)`.
            For this reason, in the :py:attr:`sampling_params` attribute, the default values are set to ``loc``:math:`=0` and ``scale``:math:`=\\sqrt{2}`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` parameter.

        .. seealso::

            - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` for Hermite ensemble
            - :cite:`DuEd02` II-C
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'banded'
        params = {'loc': loc, 'scale': scale, 'size_N': size_N}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample N i.i.d. gaussian N(mu, sigma^2)
            sampl = rng.normal(loc=params['loc'],
                               scale=params['scale'],
                               size=params['size_N'])
        else:  # if beta > 0
            sampl = rm.mu_ref_normal_sampler_tridiag(loc=params['loc'],
                                                     scale=params['scale'],
                                                     beta=self.beta,
                                                     size=params['size_N'],
                                                     random_state=rng)

        self.list_of_samples.append(sampl)

    def normalize_points(self, points):
        """ Normalize points obtained after sampling to fit the limiting distribution i.e. semi-circle

        .. math::

            f(x) = \\frac{1}{2\\pi} \\sqrt{4-x^2}

        :param points:
            A sample from Hermite ensemble, accessible through the :py:attr:`list_of_samples` attribute
        :type points:
            array_like

        - If sampled using :py:meth:`sample_banded_model` with reference measure :math:`\\mathcal{N}(\\mu,\\sigma^2)`

            1. Normalize the points to fit the p.d.f. of :math:`\\mathcal{N}(0,2)` reference measure of the :ref:`full matrix model <hermite_full_matrix_model>`

                .. math::

                    x \\mapsto \\sqrt{2}\\frac{x-\\mu}{\\sigma}

            2. If :py:attr:`beta` :math:`>0`, normalize the points to fit the semi-circle distribution

                .. math::

                    x \\mapsto \\frac{x}{\\beta N}

                Otherwise if :py:attr:`beta` :math:`=0` do nothing more

        - If sampled using :py:meth:`sample_full_model`, apply 2. above

        .. note::

            This method is called in :py:meth:`plot` and :py:meth:`hist` when ``normalization=True``
        """

        if self.sampling_mode == 'banded':
            # Normalize to fit N(0,2) ref measure of the full matrix model
            points -= self.params['loc']
            points /= np.sqrt(0.5) * self.params['scale']
        else:  # 'full'
            pass

        if self.beta > 0:
            # Normalize to fit the semi-circle distribution
            points /= np.sqrt(self.beta * self.params['size_N'])
        else:
            pass

        return points

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError('Empty `list_of_samples`,sample first!')
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample
            if normalization:
                points = self.normalize_points(points)

        fig, ax = plt.subplots(1, 1)
        # Title
        str_beta = '' if self.beta > 0 else 'with i.i.d. draws'
        title = '\n'.join([self._str_title, str_beta])
        plt.title(title)

        if self.beta == 0:
            if normalization:
                # Display N(0,2) reference measure of the full matrix model
                mu, sigma = 0.0, np.sqrt(2.0)
                x = mu + 3.5 * sigma * np.linspace(-1, 1, 100)
                ax.plot(x, sp_gaussian.pdf(x, mu, sigma),
                        'r-', lw=2, alpha=0.6,
                        label=r'$\mathcal{{N}}(0,2)$')
            else:
                pass

        else:  # self.beta > 0
            if normalization:
                # Display the limiting distribution: semi circle law
                x = np.linspace(-2, 2, 100)
                ax.plot(x, rm.semi_circle_law(x),
                        'r-', lw=2, alpha=0.6,
                        label=r'$f_{semi-circle}$')
            else:
                pass

        if display_type == 'scatter':
            ax.scatter(points, np.zeros_like(points),
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
            When ``True``, using :py:meth:`normalize_points`, display:

            - If :py:attr:`beta` :math:`=0` p.d.f. of the :math:`\\mathcal{N}(0, 2)` reference measure associated to full :ref:`full matrix model <hermite_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0` limiting distribution: semi-circle

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`hist`
            - :ref:`Full matrix model <hermite_full_matrix_model>` for Hermite ensemble
            - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` for Hermite ensemble
        """

        self.__display_and_normalization('scatter', normalization)

    def hist(self, normalization=True):
        """ Display the histogram of the last realization of the :class:`HermiteEnsemble` object.

        :param normalization:
            When ``True``, using :py:meth:`normalize_points`, display:

            - If :py:attr:`beta` :math:`=0` p.d.f. of the :math:`\\mathcal{N}(0, 2)` reference measure associated to full :ref:`full matrix model <hermite_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0` limiting distribution: semi-circle

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`plot`
            - :ref:`Full matrix model <hermite_full_matrix_model>` for Hermite ensemble
            - :ref:`Tridiagonal matrix model <hermite_banded_matrix_model>` for Hermite ensemble
        """
        self.__display_and_normalization('hist', normalization)


class LaguerreEnsemble(BetaEnsemble):
    """ Laguerre Ensemble object

    .. seealso::

        - :ref:`Full matrix model <laguerre_full_matrix_model>` for Laguerre ensemble
        - :ref:`Tridiagonal matrix model <laguerre_banded_matrix_model>` for Laguerre ensemble
    """

    def __init__(self, beta=2):

        super().__init__(beta=beta)

        params = {'shape': 0.0, 'scale': 2.0,
                  'size_N': 10, 'size_M': None}
        self.params.update(params)

    def sample_full_model(self, size_N=10, size_M=100,
                          random_state=None):
        """ Sample from :ref:`full matrix model <Laguerre_full_matrix_model>` for Laguerre ensemble. Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\Gamma(k,\\theta)` reference measure

        :param size_N:
            Number :math:`N` of points i.e. size of the matrix to be diagonalized.
            First dimension of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_full_matrix_model>`.
        :type size_N:
            int, default :math:`10`

        :param size_M:
            Second dimension :math:`M` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_full_matrix_model>`.

        :type size_M:
            int, default :math:`100`

        .. note::

            The reference measure associated with the :ref:`full matrix model <laguerre_full_matrix_model>` is :math:`\\Gamma\\left(\\frac{\\beta}{2}(M-N+1), 2\\right)`.
            For this reason, in the :py:attr:`sampling_params`, the values of the parameters are set to ``shape``:math:`=\\frac{\\beta}{2}(M-N+1)` and ``scale``:math:`=2`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` and ``size_M`` parameters.

        .. seealso::

            - :ref:`Full matrix model <Laguerre_full_matrix_model>` for Laguerre ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'full'

        if size_M >= size_N:
            # Define the parameters of the associated Gamma distribution
            shape, scale = 0.5 * self.beta * (size_M - size_N + 1), 2.0
        else:
            err_print = ('Must have M >= N.',
                         'Given: M={} < N={}'.format(size_M, size_N))
            raise ValueError(' '.join(err_print))

        params = {'shape': shape, 'scale': scale,
                  'size_N': size_N, 'size_M': size_M}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # rng.gamma(shape=0,...) when doesn't return error! contrary to sp.stats.gamma(a=0).rvs(), see https://github.com/numpy/numpy/issues/12367
            # sampl = sp_gamma.rvs(a=params['shape'], loc=0.0, scale=params['scale'],                                       size=params['size_N'])
            if params['shape'] > 0:
                sampl = rng.gamma(shape=params['shape'],
                                  scale=params['scale'],
                                  size=params['size_N'])
            else:
                err_print = ('shape<=0.',
                             'Here beta=0, hence shape=beta/2*(M-N+1)=0')
                raise ValueError(' '.join(err_print))

        else:  # if beta > 0
            sampl = rm.laguerre_sampler_full(M=params['size_M'],
                                             N=params['size_N'],
                                             beta=self.beta,
                                             random_state=rng)

        self.list_of_samples.append(sampl)

    def sample_banded_model(self,
                            shape=1.0, scale=2.0,
                            size_N=10, size_M=None,
                            random_state=None):
        """ Sample from :ref:`tridiagonal matrix model <Laguerre_full_matrix_model>` for Laguerre ensemble. Available for :py:attr:`beta` :math:`>0` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\Gamma(k,\\theta)` reference measure

        :param shape:
            Shape parameter :math:`k` of :math:`\\Gamma(k, \\theta)` reference measure
        :type shape:
            float, default :math:`1`

        :param scale:
            Scale parameter :math:`\\theta` of :math:`\\Gamma(k, \\theta)` reference measure
        :type scale:
            float, default :math:`2.0`

        :param size_N:
            Number :math:`N` of points i.e. size of the matrix to be diagonalized.
            First dimension :math:`N` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_full_matrix_model>`.
        :type size_N:
            int, default :math:`10`

        :param size_M:
            Second dimension :math:`M` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <laguerre_full_matrix_model>`.

        :type size_M:
            int, default None

        - If ``size_M`` is not provided:

            In the :py:attr:`sampling_params`, ``size_M`` is set to
            ``size_M``:math:`= \\frac{2k}{\\beta} + N - 1`, to give an idea of the corresponding second dimension :math:`M`.


        - If ``size_M`` is provided:

            In the :py:attr:`sampling_params`, ``shape`` and ``scale`` are set to:
            ``shape``:math:`=\\frac{1}{2} \\beta (M-N+1)` and ``scale``:math:`=2`


        .. note::

            The reference measure associated with the :ref:`full matrix model <laguerre_full_matrix_model>` is :math:`\\Gamma\\left(\\frac{\\beta}{2}(M-N+1), 2\\right)`. This explains the role of the ``size_M`` parameter.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` and ``size_M`` parameters

        .. seealso::

            - :ref:`Tridiagonal matrix model <Laguerre_banded_matrix_model>` for Laguerre ensemble
            - :cite:`DuEd02` III-B
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'banded'

        if not size_M:  # Default setting
            if self.beta > 0:
                size_M = 2 / self.beta * shape + size_N - 1
            else:
                size_M = np.inf

        elif size_M >= size_N:
            # define the parameters of the associated Gamma distribution
            shape, scale = 0.5 * self.beta * (size_M - size_N + 1), 2.0

        else:
            err_print = ('Must have M >= N.',
                         'Given: M={} < N={}'.format(size_M, size_N))
            raise ValueError(' '.join(err_print))

        params = {'shape': shape, 'scale': scale,
                  'size_N': size_N, 'size_M': size_M}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # rng.gamma(shape=0,...) when doesn't return error! contrary to sp.stats.gamma(a=0).rvs(), see https://github.com/numpy/numpy/issues/12367
            # sampl = sp_gamma.rvs(a=params['shape'], loc=0.0, scale=params['scale'], size=params['size_N'])
            if params['shape'] > 0:
                sampl = rng.gamma(shape=params['shape'],
                                  scale=params['scale'],
                                  size=params['size_N'])
            else:
                err_print = ('shape<=0.',
                             'Here beta=0, hence shape=beta/2*(M-N+1)=0')
                raise ValueError(' '.join(err_print))

        else:  # if beta > 0
            sampl = rm.mu_ref_gamma_sampler_tridiag(shape=params['shape'],
                                                    scale=params['scale'],
                                                    beta=self.beta,
                                                    size=params['size_N'],
                                                    random_state=rng)

        self.list_of_samples.append(sampl)

    def normalize_points(self, points):
        """ Normalize points obtained after sampling to fit the limiting distribution i.e. Marcenko Pastur law

        .. math::

            \\frac{1}{2\\pi}
            \\frac{\\sqrt{(\\lambda_+-x)(x-\\lambda_-)}}{cx}
            1_{[\\lambda_-,\\lambda_+]}
            dx

        where   :math:`c = \\frac{M}{N}` and :math:`\\lambda_\\pm = (1\\pm\\sqrt{c})^2`

        :param points:
            A sample from Laguerre ensemble, accessible through the :py:attr:`list_of_samples` attribute
        :type points:
            array_like

        - If sampled using :py:meth:`sample_banded_model` with reference measure :math:`\\Gamma(k,\\theta)`

            .. math::

                x \\mapsto \\frac{2x}{\\theta}
                \\quad \\text{and} \\quad
                x \\mapsto \\frac{x}{\\beta M}

        - If sampled using :py:meth:`sample_full_model`

            .. math::

                x \\mapsto \\frac{x}{\\beta M}

        .. note::

            This method is called in :py:meth:`plot` and :py:meth:`hist` when ``normalization=True``.
        """

        if self.sampling_mode == 'banded':
            # Normalize to fit Gamma(*,2) refe measure of the full matrix model
            points /= 0.5 * self.params['scale']
        else:  # self.sampling_mode == 'full':
            pass

        if self.beta > 0:
            # Normalize to fit the Marcenko Pastur distribution
            points /= self.beta * self.params['size_M']
        else:
            pass
            # set a warning, won't concentrate as semi circle, they are i.i.d.

        return points

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError('Empty `list_of_samples`, sample first!')
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample
            if normalization:
                points = self.normalize_points(points)

        N, M = [self.params[key] for key in ['size_N', 'size_M']]

        fig, ax = plt.subplots(1, 1)
        # Title
        str_ratio = r'with ratio $M/N \approx {:.3f}$'.format(M / N)
        # Answers Issue #33 raised by @adrienhardy
        str_beta = '' if self.beta > 0 else 'with i.i.d. draws'
        title = '\n'.join([self._str_title,
                           ' '.join([str_ratio, str_beta])])
        plt.title(title)

        if self.beta == 0:
            if normalization:
                # Display Gamma(k,2) reference measure of the full matrix model
                k, theta = self.params['shape'], 2
                x = np.linspace(0, np.max(points) + 4, 100)
                ax.plot(x, sp_gamma.pdf(x, a=k, loc=0.0, scale=theta),
                        'r-', lw=2, alpha=0.6,
                        label=r'$\Gamma({},{})$'.format(k, theta))

        else:  # self.beta > 0
            if normalization:
                # Display the limiting distribution: Marcenko Pastur
                x = np.linspace(1e-2, np.max(points) + 0.3, 100)
                ax.plot(x, rm.marcenko_pastur_law(x, M, N),
                        'r-', lw=2, alpha=0.6,
                        label=r'$f_{Marcenko-Pastur}$')

        if display_type == 'scatter':
            ax.scatter(points, np.zeros_like(points),
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
            When ``True``, using :py:meth:`normalize_points`, display:

            - If :py:attr:`beta` :math:`=0` p.d.f. of the :math:`\\Gamma(k, 2)` reference measure associated to full :ref:`full matrix model <laguerre_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0` limiting distribution: Marcenko-Pastur

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`hist`
            - :ref:`Full matrix model <Laguerre_full_matrix_model>` for Laguerre ensemble
            - :ref:`Tridiagonal matrix model <Laguerre_banded_matrix_model>` for Laguerre ensemble
        """

        self.__display_and_normalization('scatter', normalization)

    def hist(self, normalization=True):
        """ Display the histogram of the last realization of the :class:`LaguerreEnsemble` object.

        :param normalization:
            When ``True``, using :py:meth:`normalize_points`, display:

            - If :py:attr:`beta` :math:`=0` p.d.f. of the :math:`\\Gamma(k, 2)` reference measure associated to full :ref:`full matrix model <laguerre_full_matrix_model>`
            - If :py:attr:`beta` :math:`>0` limiting distribution: Marcenko-Pastur

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`plot`
            - :ref:`Full matrix model <Laguerre_full_matrix_model>` for Laguerre ensemble
            - :ref:`Tridiagonal matrix model <Laguerre_banded_matrix_model>` for Laguerre ensemble
        """
        self.__display_and_normalization('hist', normalization)


class JacobiEnsemble(BetaEnsemble):
    """ Jacobi Ensemble object

    .. seealso::

        - :ref:`Full matrix model <jacobi_full_matrix_model>` for Jacobi ensemble
        - :ref:`Tridiagonal matrix model <jacobi_banded_matrix_model>` for Jacobi ensemble
    """

    def __init__(self, beta=2):

        super().__init__(beta=beta)

        params = {'a': 1.0, 'b': 1.0,
                  'size_N': 10, 'size_M1': None, 'size_M2': None}
        self.params.update(params)

    def sample_full_model(self, size_N=100, size_M1=150, size_M2=200,
                          random_state=None):
        """ Sample from :ref:`full matrix model <Jacobi_full_matrix_model>` for Jacobi ensemble. Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\operatorname{Beta}(a,b)` reference measure

        :param size_N:
            Number :math:`N` of points i.e. size of the matrix to be diagonalized.
            First dimension of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type size_N:
            int, default :math:`100`

        :param size_M1:
            Second dimension :math:`M_1` of the first matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type size_M1:
            int, default :math:`150`

        :param size_M2:
            Second dimension :math:`M_2` of the second matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type size_M2:
            int, default :math:`200`

        .. note::

            The reference measure associated with the :ref:`full matrix model <jacobi_full_matrix_model>` is

            .. math::

                \\operatorname{Beta}\\left(\\frac{\\beta}{2}(M_1-N+1), \\frac{\\beta}{2}(M_2-N+1)\\right)

            For this reason, in the :py:attr:`sampling_params` attribute, the values of the parameters are set to ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N``, ``size_M2`` and ``size_M2`` parameters.

        .. seealso::

            - :ref:`Full matrix model <Jacobi_full_matrix_model>` for Jacobi ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'full'

        if (size_M1 >= size_N) and (size_M2 >= size_N):
            # all([var >= size_N for var in [size_M1, size_M2]]
            a = 0.5 * self.beta * (size_M1 - size_N + 1)
            b = 0.5 * self.beta * (size_M2 - size_N + 1)

        else:
            err_print = ('Must have M1, M2 >= N.',
                         'Given: M1={}, M2={} and N={}'.format(size_M1,
                                                               size_M2,
                                                               size_N))
            raise ValueError(' '.join(err_print))

        params = {'a': a, 'b': b,
                  'size_N': size_N, 'size_M1': size_M1, 'size_M2': size_M2}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample i.i.d. Beta(a,b) if size_M1,2 were used a,b = beta/2 (M_1,2 - N + 1) = 0 => ERROR
            sampl = rng.beta(a=params['a'], b=params['b'],
                             size=params['size_N'])
        else:
            sampl = rm.jacobi_sampler_full(M_1=params['size_M1'],
                                           M_2=params['size_M2'],
                                           N=params['size_N'],
                                           beta=self.beta,
                                           random_state=rng)

        self.list_of_samples.append(sampl)

    def sample_banded_model(self, a=1.0, b=2.0,
                            size_N=10, size_M1=None, size_M2=None,
                            random_state=None):
        """ Sample from :ref:`tridiagonal matrix model <Jacobi_full_matrix_model>` for Jacobi ensemble. Available for :py:attr:`beta` :math:`>0` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. points from the :math:`\\operatorname{Beta}(a,b)` reference measure

        :param shape:
            Shape parameter :math:`k` of :math:`\\Gamma(k, \\theta)` reference measure
        :type shape:
            float, default :math:`1`

        :param scale:
            Scale parameter :math:`\\theta` of :math:`\\Gamma(k, \\theta)` reference measure
        :type scale:
            float, default :math:`2.0`

        :param size_N:
            Number :math:`N` of points i.e. size of the matrix to be diagonalized.
            First dimension :math:`N` of the matrix used to form the covariance matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type size_N:
            int, default :math:`10`

        :param size_M1:
            Second dimension :math:`M_1` of the first matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type size_M1:
            int, default :math:`150`

        :param size_M2:
            Second dimension :math:`M_2` of the second matrix used to form the matrix to be diagonalized, see :ref:`full matrix model <jacobi_full_matrix_model>`.
        :type size_M2:
            int, default :math:`200`

        .. note::

            The reference measure associated with the :ref:`full matrix model <jacobi_full_matrix_model>` is :

            .. math::

                \\operatorname{Beta}\\left(\\frac{\\beta}{2}(M_1-N+1), \\frac{\\beta}{2}(M_2-N+1)\\right)

            For this reason, in the :py:attr:`sampling_params` attribute, the values of the parameters are set to ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N``, ``size_M2`` and ``size_M2`` parameters.

        - If ``size_M1`` and ``size_M2`` are not provided:

            In the :py:attr:`sampling_params` attribute, ``size_M1,2`` are set to
            ``size_M1``:math:`= \\frac{2a}{\\beta} + N - 1` and ``size_M2``:math:`= \\frac{2b}{\\beta} + N - 1`, to give an idea of the corresponding second dimensions :math:`M_{1,2}`.

        - If ``size_M1`` and ``size_M2`` are provided:

            In the :py:attr:`sampling_params` attribute, ``a`` and ``b`` are set to:
            ``a``:math:`=\\frac{\\beta}{2}(M_1-N+1)` and
            ``b``:math:`=\\frac{\\beta}{2}(M_2-N+1)`.

        .. seealso::

            - :ref:`Tridiagonal matrix model <Jacobi_banded_matrix_model>` for Jacobi ensemble
            - :cite:`KiNe04` Theorem 2
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'banded'

        if not (size_M1 and size_M2):  # default setting

            if self.beta > 0:
                size_M1 = 2 / self.beta * a + size_N - 1
                size_M2 = 2 / self.beta * b + size_N - 1

            else:
                size_M1, size_M2 = np.inf, np.inf

        elif (size_M1 >= size_N) and (size_M2 >= size_N):
            # all([var >= size_N for var in [size_M1, size_M2]]
            a = 0.5 * self.beta * (size_M1 - size_N + 1)
            b = 0.5 * self.beta * (size_M2 - size_N + 1)

        else:
            err_print = ('Must have M1, M2 >= N.',
                         'Given: M1={}, M2={} and N={}'.format(size_M1,
                                                               size_M2,
                                                               size_N))
            raise ValueError(' '.join(err_print))

        params = {'a': a, 'b': b,
                  'size_N': size_N, 'size_M1': size_M1, 'size_M2': size_M2}
        self.params.update(params)

        if self.beta == 0:  # Answer issue #28 raised by @rbardenet
            # Sample i.i.d. Beta(a,b)
            # If size_M1,2 is used a, b = beta/2 (M1,2 - N + 1) = 0 => ERROR
            sampl = rng.beta(a=params['a'], b=params['b'],
                             size=params['size_N'])
        else:
            sampl = rm.mu_ref_beta_sampler_tridiag(a=params['a'],
                                                   b=params['b'],
                                                   beta=self.beta,
                                                   size=params['size_N'],
                                                   random_state=rng)

        self.list_of_samples.append(sampl)

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError('Empty `list_of_samples`, sample first!')
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample

        N, M_1, M_2 = [self.params[key]
                       for key in ['size_N', 'size_M1', 'size_M2']]

        fig, ax = plt.subplots(1, 1)
        # Title, answers Issue #33 raised by @adrienhardy
        str_ratio = ', '.join(['with ratios',
                               r'$M_1/N \approx {:.3f}$'.format(M_1 / N),
                               r'$M_2/N \approx {:.3f}$'.format(M_2 / N)])
        str_beta = '' if self.beta > 0 else 'with i.i.d. draws'
        title = '\n'.join([self._str_title, ' '.join([str_ratio, str_beta])])
        plt.title(title)

        if self.beta == 0:
            if normalization:
                # Display Beta(a,b) reference measure
                a, b = [self.params[key] for key in ['a', 'b']]
                x = np.linspace(0, 1, 100)
                ax.plot(x, sp_beta.pdf(x, a=a, b=b),
                        'r-', lw=2, alpha=0.6,
                        label=r'$\operatorname{{Beta}}({},{})$'.format(a, b))
        else:  # self.beta > 0
            if normalization:
                # Display the limiting distribution: Wachter law
                eps = 5e-3
                x = np.linspace(eps, 1.0 - eps, 500)
                ax.plot(x, rm.wachter_law(x, M_1, M_2, N),
                        'r-', lw=2, alpha=0.6,
                        label=r'$f_{Wachter}$')

        if display_type == 'scatter':
            ax.scatter(points, np.zeros_like(points),
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
            When ``True``, display:

            - If :py:attr:`beta` :math:`=0` p.d.f. of the :math:`\\operatorname{Beta}(a, b)`
            - If :py:attr:`beta` :math:`>0` limiting distribution: Wachter

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`hist`
            - :ref:`Full matrix model <Jacobi_full_matrix_model>` for Jacobi ensemble
            - :ref:`Tridiagonal matrix model <Jacobi_banded_matrix_model>` for Jacobi ensemble
        """

        self.__display_and_normalization('scatter', normalization)

    def hist(self, normalization=True):
        """ Display the histogram of the last realization of the :class:`JacobiEnsemble` object.

        :param normalization:
            When ``True``, display:

            - If :py:attr:`beta` :math:`=0` p.d.f. of the :math:`\\operatorname{Beta}(a, b)`
            - If :py:attr:`beta` :math:`>0` limiting distribution: Wachter

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`normalize_points`
            - :py:meth:`plot`
            - :ref:`Full matrix model <Jacobi_full_matrix_model>` for Jacobi ensemble
            - :ref:`Tridiagonal matrix model <Jacobi_banded_matrix_model>` for Jacobi ensemble
        """
        self.__display_and_normalization('hist', normalization)


class CircularEnsemble(BetaEnsemble):
    """ Circular Ensemble object

    .. seealso::

        - :ref:`Full matrix model <circular_full_matrix_model>` for Circular ensemble
        - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` for Circular ensemble
    """

    def __init__(self, beta=2):

        if not isinstance(beta, int):
            raise ValueError('`beta` must be int >0. Given: {}'.format(beta))
        super().__init__(beta=beta)

        params = {'size_N': 10}
        self.params.update(params)

    def sample_full_model(self, size_N=10, haar_mode='Hermite',
                          random_state=None):
        """ Sample from :ref:`tridiagonal matrix model <Circular_full_matrix_model>` for Circular ensemble. Only available for :py:attr:`beta` :math:`\\in\\{1, 2, 4\\}` and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. uniform points on the unit circle

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

            - :ref:`Full matrix model <circular_full_matrix_model>` for Circular ensemble
            - :py:meth:`sample_banded_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'full'
        params = {'size_N': size_N, 'haar_mode': haar_mode}
        self.params.update(params)

        if self.beta == 0:  # i.i.d. points uniformly on the circle
            # Answer issue #28 raised by @rbardenet
            sampl = np.exp(2 * 1j * np.pi * rng.rand(params['size_N']))
        else:
            sampl = rm.circular_sampler_full(N=params['size_N'],
                                             beta=self.beta,
                                             haar_mode=params['haar_mode'],
                                             random_state=rng)

        self.list_of_samples.append(sampl)

    def sample_banded_model(self, size_N=10,
                            random_state=None):
        """ Sample from :ref:`tridiagonal matrix model <Circular_full_matrix_model>` for Circular Ensemble.
        Available for :py:attr:`beta` :math:`\\in\\mathbb{N}^*`, and the degenerate case :py:attr:`beta` :math:`=0` corresponding to i.i.d. uniform points on the unit circle

        :param size_N:
            Number :math:`N` of points i.e. size of the matrix to be diagonalized
        :type size_N:
            int, default :math:`10`

        .. note::

            To compare :py:meth:`sample_banded_model` with :py:meth:`sample_full_model` simply use the ``size_N`` parameter.

        .. seealso::

            - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` for Circular ensemble
            - :py:meth:`sample_full_model`
        """
        rng = check_random_state(random_state)

        self.sampling_mode = 'banded'
        params = {'size_N': size_N}
        self.params.update(params)

        if self.beta == 0:  # i.i.d. points uniformly on the circle
            # Answer issue #28 raised by @rbardenet
            sampl = np.exp(2 * 1j * np.pi * rng.rand(params['size_N']))
        else:
            sampl = rm.mu_ref_unif_unit_circle_sampler_quindiag(
                        beta=self.beta,
                        size=params['size_N'],
                        random_state=rng)

        self.list_of_samples.append(sampl)

    def __display_and_normalization(self, display_type):

        if not self.list_of_samples:
            raise ValueError('Empty `list_of_samples`, sample first!')
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample

        fig, ax = plt.subplots(1, 1)

        # Title
        samp_mod = ''

        if self.beta == 0:
            samp_mod = r'i.i.d samples from $\mathcal{U}_{[0,2\pi]}$'
        elif self.sampling_mode == 'full':
            samp_mod = 'full matrix model with haar_mode={}'.format(
                self.params['haar_mode'])
        else:  # self.sampling_mode == 'banded':
            samp_mod = 'quindiag model'

        title = '\n'.join([self._str_title, 'using {}'.format(samp_mod)])
        plt.title(title)

        if display_type == 'scatter':
            # Draw unit circle
            unit_circle = plt.Circle((0, 0), 1, color='r', fill=False)
            ax.add_artist(unit_circle)

            ax.set_xlim([-1.3, 1.3])
            ax.set_ylim([-1.3, 1.3])
            ax.set_aspect('equal')
            ax.scatter(points.real, points.imag, c='blue', label='sample')

        elif display_type == 'hist':
            points = np.angle(points)

            # Uniform distribution on [0, 2pi]
            ax.axhline(y=1 / (2 * np.pi), color='r', label=r'$\frac{1}{2\pi}$')

            ax.hist(points, bins=30, density=1,
                    facecolor='blue', alpha=0.5,
                    label='hist')
        else:
            pass

        plt.legend(loc='best', frameon=False)

    def plot(self):
        """ Display the last realization of the :class:`CircularEnsemble` object

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`hist`
            - :ref:`Full matrix model <circular_full_matrix_model>` for Circular ensemble
            - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` for Circular ensemble
        """

        self.__display_and_normalization('scatter')

    def hist(self):
        """ Display the histogram of the last realization of the :class:`CircularEnsemble` object.

        .. seealso::

            - :py:meth:`sample_full_model`, :py:meth:`sample_banded_model`
            - :py:meth:`plot`
            - :ref:`Full matrix model <circular_full_matrix_model>` for Circular ensemble
            - :ref:`Quindiagonal matrix model <circular_banded_matrix_model>` for Circular ensemble
        """
        self.__display_and_normalization('hist')


class GinibreEnsemble(BetaEnsemble):
    """ Ginibre Ensemble object

    .. seealso::

        - :ref:`Full matrix model <ginibre_full_matrix_model>` for Ginibre ensemble
    """

    def __init__(self, beta=2):

        if beta != 2:
            err_print = ('Ginibre ensemble is only available for `beta`=2.',
                         'Given {}'.format(beta))
            raise ValueError(' '.join(err_print))
        super().__init__(beta=beta)

        params = {'size_N': 10}
        self.params.update(params)

    def sample_full_model(self, size_N=10,
                          random_state=None):
        """ Sample from :ref:`full matrix model <Ginibre_full_matrix_model>` for Ginibre ensemble. Only available for :py:attr:`beta` :math:`=2`

        :param size_N:
            Number :math:`N` of points i.e. size of the matrix to be diagonalized
        :type size_N:
            int, default :math:`10`

        .. seealso::

            - :ref:`Full matrix model <ginibre_full_matrix_model>` for Ginibre ensemble
        """
        rng = check_random_state(random_state)

        self.params.update({'size_N': size_N})
        sampl = rm.ginibre_sampler_full(N=self.params['size_N'],
                                        random_state=rng)

        self.list_of_samples.append(sampl)

    def sample_banded_model(self, *args):
        """ No banded model is known for Ginibre, use :py:meth:`sample_full_model`
        """
        raise NotImplementedError('No banded model is known for Ginibre, use `sample_full_model`')

    def normalize_points(self, points):

        return points / np.sqrt(self.params['size_N'])

    def __display_and_normalization(self, display_type, normalization):

        if not self.list_of_samples:
            raise ValueError('Empty `list_of_samples`, sample first!')
        else:
            points = self.list_of_samples[-1].copy()  # Pick last sample
            if normalization:
                points = self.normalize_points(points)

        fig, ax = plt.subplots(1, 1)

        title = self._str_title

        if display_type == 'scatter':

            if normalization:
                # Draw unit circle
                unit_circle = plt.Circle((0, 0), 1, color='r', fill=False)
                ax.add_artist(unit_circle)

                ax.set_xlim([-1.3, 1.3])
                ax.set_ylim([-1.3, 1.3])
                ax.set_aspect('equal')

            ax.scatter(points.real, points.imag, c='blue', label='sample')

        elif display_type == 'hist':
            title = '\n'.join([title,
                               'Histogram of the modulus of each points'])

            if normalization:
                ax.plot([0, 1, 1], [0, 2, 0], color='r')

            bins = np.linspace(0, 1, 20)
            ax.hist(np.abs(points), bins=bins, density=1,
                    facecolor='blue', alpha=0.5,
                    label='hist')
        else:
            pass

        plt.title(title)

        # plt.legend(loc='best', frameon=False)

    def plot(self, normalization=True):
        """ Display the last realization of the :class:`GinibreEnsemble` object

        :param normalization:
            When ``True``, the points are normalized so as to concentrate in the unit disk.

            .. math::

                x \\mapsto \\frac{x}{\\sqrt{N}}

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`
            - :ref:`Full matrix model <ginibre_full_matrix_model>` for Ginibre ensemble ensemble
        """

        self.__display_and_normalization('scatter', normalization)

    def hist(self, normalization=True):
        """ Display the histogram of the radius of the points the last realization of the :class:`GinibreEnsemble` object

        :param normalization:
            When ``True``, the points are normalized so as to concentrate in the unit disk.

            .. math::

                x \\mapsto \\frac{x}{\\sqrt{N}}

        :type normalization:
            bool, default ``True``

        .. seealso::

            - :py:meth:`sample_full_model`
            - :ref:`Full matrix model <ginibre_full_matrix_model>` for Ginibre ensemble ensemble
        """

        self.__display_and_normalization('hist', normalization)
