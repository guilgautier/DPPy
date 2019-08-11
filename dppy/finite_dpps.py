# coding: utf8
""" Implementation of :py:class:`FiniteDPP` object which has 6 main methods:

- :py:meth:`~FiniteDPP.sample_exact`, see also :ref:`sampling DPPs exactly<finite_dpps_exact_sampling>`
- :py:meth:`~FiniteDPP.sample_exact_k_dpp`, see also :ref:`sampling k-DPPs exactly<finite_dpps_exact_sampling>`
- :py:meth:`~FiniteDPP.sample_mcmc`, see also :ref:`sampling DPPs with MCMC<finite_dpps_mcmc_sampling>`
- :py:meth:`~FiniteDPP.sample_mcmc_k_dpps`, see also :ref:`sampling k-DPPs with MCMC<finite_dpps_mcmc_sampling_k_dpps>`
- :py:meth:`~FiniteDPP.compute_K`, to compute the correlation :math:`K` kernel from initial parametrization
- :py:meth:`~FiniteDPP.compute_L`, to compute the likelihood :math:`L` kernel from initial parametrization

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/index.html>`_
"""

import numpy as np
import scipy.linalg as la

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

from warnings import warn

from dppy.exact_sampling import (dpp_sampler_generic_kernel,
                                 proj_dpp_sampler_kernel,
                                 proj_dpp_sampler_eig,
                                 dpp_eig_vecs_selector,
                                 dpp_eig_vecs_selector_L_dual,
                                 k_dpp_eig_vecs_selector,
                                 elementary_symmetric_polynomials)

from dppy.mcmc_sampling import dpp_sampler_mcmc, zonotope_sampler

from dppy.utils import (check_random_state,
                        is_symmetric,
                        is_projection,
                        is_orthonormal,
                        is_full_row_rank,
                        is_in_01,
                        is_geq_0,
                        is_equal_to_O_or_1)


class FiniteDPP:
    """ Finite DPP object parametrized by

    :param kernel_type:

        - ``'correlation'`` :math:`\\mathbf{K}` kernel
        - ``'likelihood'`` :math:`\\mathbf{L}` kernel

    :type kernel_type:
        string

    :param projection:
        Indicate whether the provided kernel is of projection type. This may be useful when the :class:`FiniteDPP` object is defined through its correlation kernel :math:`\\mathbf{K}`.

    :type projection:
        bool, default ``False``

    :param params:
        Dictionary containing the parametrization of the underlying

        - correlation kernel

            - ``{'K': K}``, with :math:`0 \\preceq \\mathbf{K} \\preceq I`
            - ``{'K_eig_dec': (eig_vals, eig_vecs)}``, with :math:`0 \\leq eigvals \\leq 1`
            - ``{'A_zono': A}``, with :math:`A (d \\times N)` and :math:`\\operatorname{rank}(A)=d`

        - likelihood kernel

            - ``{'L': L}``, with :math:`\\mathbf{L}\\succeq 0`
            - ``{'L_eig_dec': (eig_vals, eig_vecs)}``, with :math:`eigvals \\geq 0`
            - ``{'L_gram_factor': Phi}``, with :math:`\\mathbf{L} = \\Phi^{ \\top} \\Phi`

    :type params:
        dict

    .. caution::

        For now we only consider real valued matrices :math:`\\mathbf{K}, \\mathbf{L}, A, \\Phi`.

    .. seealso::

        - :ref:`finite_dpps_definition`
        - :ref:`finite_dpps_exact_sampling`
    """

    ###############
    # Constructor #
    ###############
    def __init__(self, kernel_type, projection=False, **params):

        self.kernel_type = kernel_type
        self.projection = projection
        self.params_keys = set(params.keys())
        self.__check_args_coherence()

        # Sampling
        self.sampling_mode = 'GS'  # Gram-Schmidt
        self.list_of_samples = []

        # when using .sample_k_dpp_*
        self.size_k_dpp = None
        self.E_poly = None  # evaluation of the

        # Attributes relative to K correlation kernel:
        # K, K_eig_vals, K_eig_vecs, A_zono
        self.K = is_symmetric(params.get('K', None))
        if self.projection:
            self.K = is_projection(self.K)

        e_vals, e_vecs = params.get('K_eig_dec', [None, None])
        if self.projection:
            self.K_eig_vals = is_equal_to_O_or_1(e_vals)
        else:
            self.K_eig_vals = is_in_01(e_vals)
        self.eig_vecs = is_orthonormal(e_vecs)

        self.A_zono = is_full_row_rank(params.get('A_zono', None))

        # Attributes relative to L likelihood kernel:
        # L, L_eig_vals, L_eig_vecs, L_gram_factor, L_dual, L_dual_eig_vals, L_dual_eig_vecs
        self.L = is_symmetric(params.get('L', None))
        if self.projection:
            self.L = is_projection(self.L)

        e_vals, e_vecs = params.get('L_eig_dec', [None, None])
        if self.projection:
            self.L_eig_vals = is_equal_to_O_or_1(e_vals)
        else:
            self.L_eig_vals = is_geq_0(e_vals)
        if self.eig_vecs is None:  # K_eig_vecs = L_eig_vecs
            self.eig_vecs = is_orthonormal(e_vecs)

        # L' "dual" likelihood kernel, L' = Phi Phi.T, Phi = L_gram_factor
        self.L_gram_factor = params.get('L_gram_factor', None)
        self.L_dual = None
        self.L_dual_eig_vals = None
        self.L_dual_eig_vecs = None

        if self.L_gram_factor is not None:
            Phi = self.L_gram_factor
            d, N = Phi.shape
            if d < N:
                self.L_dual = Phi.dot(Phi.T)
                print('L_dual = Phi Phi.T was computed: Phi (dxN) with d<N')
            else:
                if self.L is None:
                    self.L = Phi.T.dot(Phi)
                    print('L = Phi.T Phi was computed: Phi (dxN) with d>=N')

    def __str__(self):
        str_info = ['DPP defined through {} {} kernel'
                    .format('projection' if self.projection else '',
                            self.kernel_type),
                    'Parametrized by {}'
                    .format(self.params_keys),
                    '- sampling mode = {}'
                    .format(self.sampling_mode),
                    '- number of samples = {}'
                    .format(len(self.list_of_samples))]

        return '\n'.join(str_info)

    # Check routine
    def __check_args_coherence(self):
        # Check coherence of initialization parameters of the DPP:
        # kernel_type, projection and params.

        K_type, K_params = 'correlation', {'K', 'K_eig_dec', 'A_zono'}
        L_type, L_params = 'likelihood', {'L', 'L_eig_dec', 'L_gram_factor'}

        if self.kernel_type == K_type:
            if self.params_keys.intersection(K_params):
                if 'A_zono' in self.params_keys and not self.projection:
                    warn('Weird setting: correlation kernel defined via `A_zono` but `projection`=False. `projection` switched to True')
                    self.projection = True
            else:
                err_print =\
                    ['Invalid parametrization of correlation kernel, choose:',
                     '- `K` = 0 <= K <= I',
                     '- `K_eig_dec` = (eig_vals, eig_vecs) 0 <= eig_vals <= 1',
                     '- `A_zono` = A is dxN matrix, K = A.T (AA.T)^-1 A',
                     'Given: {}'.format(self.params_keys)]
                raise ValueError('\n'.join(err_print))

        elif self.kernel_type == L_type:
            if self.params_keys.intersection(L_params):
                if self.projection:
                    warn('weird setting: defining a DPP via a projection likelihood L kernel is unusual. Make sure you do not want to use a projection CORRELATION K kernel instead')
            else:
                err_print =\
                    ['Invalid parametrization of likelihood kernel, choose:',
                     '- `L` = L >= 0',
                     '- `L_eig_dec` = (eig_vals, eig_vecs) eig_vals >= 0',
                     '- `L_gram_factor` = Phi (dxN) where L = Phi.T Phi',
                     'Given: {}'.format(self.params_keys)]
                raise ValueError('\n'.join(err_print))

        else:
            err_print =\
                ['Invalid `kernel_type`, choose among:',
                 '- `correlation`: K kernel',
                 '- `likelihood`: L kernel'
                 'Given: {}'.format(self.params_keys)]
            raise ValueError('\n'.join(err_print))

    ##################
    # Object methods #
    ##################
    def info(self):
        """ Display infos about the :class:`FiniteDPP` object
        """
        print(self.__str__())

    def flush_samples(self):
        """ Empty the :py:attr:`~FiniteDPP.list_of_samples`.

        .. see also::

            - :py:meth:`~FiniteDPP.sample_exact`
            - :py:meth:`~FiniteDPP.sample_mcmc`
        """
        self.list_of_samples = []

    # Exact sampling
    def sample_exact(self, mode='GS', random_state=None):
        """ Sample exactly from the corresponding :class:`FiniteDPP <FiniteDPP>` object. The sampling scheme is based on the chain rule with Gram-Schmidt like updates of the conditionals.

        :param mode:

            - ``projection=True``:
                - ``'GS'`` (default): Gram-Schmidt on the rows of :math:`\\mathbf{K}`.
                - ``'Chol'`` :cite:`Pou19` Algorithm 3
                - ``'Schur'``: when DPP defined from correlation kernel ``K``, use Schur complement to compute conditionals

            - ``projection=False``:
                - ``'GS'`` (default): Gram-Schmidt on the rows of the eigenvectors of :math:`\\mathbf{K}` selected in Phase 1.
                - ``'GS_bis'``: Slight modification of ``'GS'``
                - ``'Chol'`` :cite:`Pou19` Algorithm 1
                - ``'KuTa12'``: Algorithm 1 in :cite:`KuTa12`
        :type mode:
            string, default ``'GS'``

        :return:
            A sample from the corresponding :class:`FiniteDPP <FiniteDPP>` object.
        :rtype:
            array_like

        .. note::

            Each time you call this function, the sample is added to the :py:attr:`~FiniteDPP.list_of_samples`.

            The latter can be emptied using :py:meth:`~FiniteDPP.flush_samples`

        .. caution::

            The underlying kernel :math:`\\mathbf{K}`, resp. :math:`\\mathbf{L}` must be real valued for now.

        .. seealso::

            - :ref:`finite_dpps_exact_sampling`
            - :py:meth:`~FiniteDPP.flush_samples`
            - :py:meth:`~FiniteDPP.sample_mcmc`
        """

        rng = check_random_state(random_state)

        self.sampling_mode = mode

        if self.sampling_mode == 'Chol':
            self.compute_K()
            if self.projection:
                sampl = proj_dpp_sampler_kernel(self.K, self.sampling_mode,
                                                random_state=rng)
            else:
                sampl, _ = dpp_sampler_generic_kernel(self.K, random_state=rng)
            self.list_of_samples.append(sampl)

        # If eigen decoposition of K, L or L_dual is available USE IT!
        elif self.K_eig_vals is not None:
            # Phase 1
            if self.kernel_type == 'correlation' and self.projection:
                V = self.eig_vecs[:, self.K_eig_vals > 0.5]
            else:
                V = dpp_eig_vecs_selector(self.K_eig_vals, self.eig_vecs,
                                          random_state=rng)
            # Phase 2
            if V.shape[1]:
                sampl = proj_dpp_sampler_eig(V, self.sampling_mode,
                                             random_state=rng)
            else:
                sampl = np.array([])
            self.list_of_samples.append(sampl)

        elif self.L_eig_vals is not None:
            self.K_eig_vals = self.L_eig_vals / (1.0 + self.L_eig_vals)
            self.sample_exact(mode=self.sampling_mode, random_state=rng)

        elif self.L_dual_eig_vals is not None:
            # Phase 1
            V = dpp_eig_vecs_selector_L_dual(self.L_dual_eig_vals,
                                             self.L_dual_eig_vecs,
                                             self.L_gram_factor,
                                             random_state=rng)
            # Phase 2
            sampl = proj_dpp_sampler_eig(V, self.sampling_mode,
                                         random_state=rng)
            self.list_of_samples.append(sampl)

        # If DPP defined via projection correlation kernel K
        # no eigendecomposition required
        elif (self.K is not None) and self.projection:
            sampl = proj_dpp_sampler_kernel(self.K, self.sampling_mode,
                                            random_state=rng)
            self.list_of_samples.append(sampl)

        elif self.L_dual is not None:
            self.L_dual_eig_vals, self.L_dual_eig_vecs =\
                la.eigh(self.L_dual)
            self.sample_exact(mode=self.sampling_mode, random_state=rng)

        elif self.K is not None:
            self.K_eig_vals, self.eig_vecs = la.eigh(self.K)
            self.sample_exact(mode=self.sampling_mode, random_state=rng)

        elif self.L is not None:
            self.L_eig_vals, self.eig_vecs = la.eigh(self.L)
            self.sample_exact(mode=self.sampling_mode, random_state=rng)

        # If DPP defined through correlation kernel with parameter 'A_zono'
        # a priori you wish to use the zonotope approximate sampler
        elif self.A_zono is not None:
            warn('DPP defined via `A_zono`, apriori you want to use `sample_mcmc`, but you have called `sample_exact`')

            self.K_eig_vals = np.ones(self.A_zono.shape[0])
            self.eig_vecs, _ = la.qr(self.A_zono.T, mode='economic')

            self.sample_exact(self.sampling_mode, random_state=rng)

    def sample_exact_k_dpp(self, size, mode='GS', random_state=None):
        """ Sample exactly from :math:`\\operatorname{k-DPP}`.
        A priori the :class:`FiniteDPP <FiniteDPP>` object was instanciated by its likelihood :math:`\\mathbf{L}` kernel so that

        .. math::

            \\mathbb{P}_{\\operatorname{k-DPP}}(\\mathcal{X} = S)
                \\propto \\det \\mathbf{L}_S ~ 1_{|S|=k}

        :param size:
            size :math:`k` of the :math:`\\operatorname{k-DPP}`
        :type size:
            int

        :param mode:
            - ``projection=True``:
                - ``'GS'`` (default): Gram-Schmidt on the rows of :math:`\\mathbf{K}`.
                - ``'Schur'``: Use Schur complement to compute conditionals.

            - ``projection=False``:
                - ``'GS'`` (default): Gram-Schmidt on the rows of the eigenvectors of :math:`\\mathbf{K}` selected in Phase 1.
                - ``'GS_bis'``: Slight modification of ``'GS'``
                - ``'KuTa12'``: Algorithm 1 in :cite:`KuTa12`
        :type mode:
            string, default ``'GS'``

        :return:
            A sample from the corresponding :math:`\\operatorname{k-DPP}`
        :rtype:
            array_like

        .. note::

            Each time you call this function, the sample is added to the :py:attr:`~FiniteDPP.list_of_samples`.

            The latter can be emptied using :py:meth:`~FiniteDPP.flush_samples`

        .. caution::

            The underlying kernel :math:`\\mathbf{K}`, resp. :math:`\\mathbf{L}` must be real valued for now.

        .. seealso::

            - :py:meth:`~FiniteDPP.sample_exact`
            - :py:meth:`~FiniteDPP.sample_mcmc_k_dpp`
        """

        rng = check_random_state(random_state)

        self.sampling_mode = mode

        # If DPP defined via projection kernel
        if self.projection:
            if self.kernel_type == 'correlation':

                if self.K_eig_vals is not None:
                    rank = np.round(np.sum(self.K_eig_vals)).astype(int)
                elif self.A_zono is not None:
                    rank = self.A_zono.shape[0]
                else: # self.K is not None
                    rank = np.round(np.trace(self.K)).astype(int)

                if size != rank:
                    raise ValueError('size k={} != rank={} for projection correlation K kernel'.format(k, rank))

                if self.K_eig_vals is not None:
                    # K_eig_vals > 0.5 below to get indices where e_vals = 1
                    sampl = proj_dpp_sampler_eig(
                            eig_vecs=self.eig_vecs[:, self.K_eig_vals > 0.5],
                            mode=self.sampling_mode,
                            size=size,
                            random_state=rng)

                elif self.A_zono is not None:
                    warn('DPP defined via `A_zono`, apriori you want to use `sampl_mcmc`, but you have called `sample_exact`')

                    self.K_eig_vals = np.ones(rank)
                    self.eig_vecs, _ = la.qr(self.A_zono.T, mode='economic')

                    sampl = proj_dpp_sampler_eig(eig_vecs=self.eig_vecs,
                                                 mode=self.sampling_mode,
                                                 size=size,
                                                 random_state=rng)

                else:
                    sampl = proj_dpp_sampler_kernel(kernel=self.K,
                                                    mode=self.sampling_mode,
                                                    size=size,
                                                    random_state=rng)

            else:  # self.kernel_type == 'likelihood':
                if self.L_eig_vals is not None:
                    # L_eig_vals > 0.5 below to get indices where e_vals = 1
                    sampl = proj_dpp_sampler_eig(
                            eig_vecs=self.eig_vecs[:, self.L_eig_vals > 0.5],
                            mode=self.sampling_mode,
                            size=size,
                            random_state=rng)
                else:
                    self.compute_L()
                    # size > rank treated internally in proj_dpp_sampler_kernel
                    sampl = proj_dpp_sampler_kernel(self.L,
                                                    mode=self.sampling_mode,
                                                    size=size,
                                                    random_state=rng)

            self.size_k_dpp = size
            self.list_of_samples.append(sampl)

        # If eigen decoposition of K, L or L_dual is available USE IT!
        elif self.L_eig_vals is not None:

            # Phase 1
            # Precompute elementary symmetric polynomials
            if self.E_poly is None or self.size_k_dpp < size:
                self.E_poly = elementary_symmetric_polynomials(self.L_eig_vals,
                                                               size)
            # Select eigenvectors
            V = k_dpp_eig_vecs_selector(self.L_eig_vals, self.eig_vecs,
                                        size=size,
                                        E_poly=self.E_poly,
                                        random_state=rng)
            # Phase 2
            self.size_k_dpp = size
            sampl = proj_dpp_sampler_eig(V, self.sampling_mode,
                                         random_state=rng)
            self.list_of_samples.append(sampl)

        elif self.L_dual_eig_vals is not None:
            # There is
            self.L_eig_vals = self.L_dual_eig_vals
            self.eig_vecs =\
                self.L_gram_factor.T.dot(self.L_dual_eig_vecs\
                                        / np.sqrt(self.L_dual_eig_vals))
            self.sample_exact_k_dpp(size, self.sampling_mode,
                                    random_state=rng)

        elif self.K_eig_vals is not None:
            np.seterr(divide='raise')
            self.L_eig_vals = self.K_eig_vals / (1.0 - self.K_eig_vals)
            self.sample_exact_k_dpp(size, self.sampling_mode,
                                    random_state=rng)

        # Otherwise eigendecomposition is necessary
        elif self.L_dual is not None:
            self.L_dual_eig_vals, self.L_dual_eig_vecs =\
                la.eigh(self.L_dual)
            self.sample_exact_k_dpp(size, self.sampling_mode,
                                    random_state=rng)

        elif self.K is not None:
            self.K_eig_vals, self.eig_vecs = la.eigh(self.K)
            self.sample_exact_k_dpp(size, self.sampling_mode,
                                    random_state=rng)

        elif self.L is not None:
            self.L_eig_vals, self.eig_vecs = la.eigh(self.L)
            self.sample_exact_k_dpp(size, self.sampling_mode,
                                    random_state=rng)

    # Approximate sampling
    def sample_mcmc(self, mode, **params):
        """ Run a MCMC with stationary distribution the corresponding :class:`FiniteDPP <FiniteDPP>` object.

        :param mode:

            - ``'AED'`` Add-Exchange-Delete
            - ``'AD'`` Add-Delete
            - ``'E'`` Exchange
            - ``'zonotope'`` Zonotope sampling

        :type mode:
            string

        :param params:
            Dictionary containing the parameters for MCMC samplers with keys

            ``'random_state'`` (default None)

            - If ``mode='AED','AD','E'``

                + ``'s_init'`` (default None) Starting state of the Markov chain
                + ``'nb_iter'`` (default 10) Number of iterations of the chain
                + ``'T_max'`` (default None) Time horizon
                + ``'size'`` (default None) Size of the initial sample for ``mode='AD'/'E'``

                    * :math:`\\operatorname{rank}(\\mathbf{K})=\\operatorname{trace}(\\mathbf{K})` for projection :math:`\\mathbf{K}` (correlation) kernel and ``mode='E'``

            - If ``mode='zonotope'``:

                + ``'lin_obj'`` linear objective in main optimization problem (default np.random.randn(N))
                + ``'x_0'`` initial point in zonotope (default A*u, u~U[0,1]^n)
                + ``'nb_iter'`` (default 10) Number of iterations of the chain
                + ``'T_max'`` (default None) Time horizon

        :type params:
            dict

        :return:
            A sample from the corresponding :class:`FiniteDPP <FiniteDPP>` object.
        :rtype:
            list of lists

        .. seealso::

            - :ref:`finite_dpps_mcmc_sampling`
            - :py:meth:`~FiniteDPP.sample_exact`
            - :py:meth:`~FiniteDPP.flush_samples`
        """

        self.sampling_mode = mode

        if self.sampling_mode == 'zonotope':
            if self.A_zono is not None:
                chain = zonotope_sampler(self.A_zono, **params)
            else:
                err_print = ['Invalid `mode=zonotope` parameter.',
                             'DPP must be defined via `A_zono`',
                             'Given: {}'.format(self.params_keys)]
                raise ValueError(' '.join(err_print))

        elif self.sampling_mode == 'E':
            if (self.kernel_type == 'correlation') and self.projection:
                self.compute_K()
                size = params.get('size', None)
                rank = np.round(np.trace(self.K)).astype(int)
                # |sample| = Tr(K) = rank(K) a.s. for projection DPP(K)
                if size == rank:
                    chain = dpp_sampler_mcmc(self.K,
                                             self.sampling_mode,
                                             **params)
                else:
                    raise ValueError('size={} != rank={} for projection correlation K kernel'.format(size, rank))
            else:
                self.compute_L()
                chain = dpp_sampler_mcmc(self.L, self.sampling_mode, **params)

        elif self.sampling_mode in ('AED', 'AD'):
            self.compute_L()
            chain = dpp_sampler_mcmc(self.L, self.sampling_mode, **params)

        else:
            err_print = ['Invalid `mode` parameter, choose among:',
                         '- `AED`: Add-Exchange-Delete',
                         '- `AD`: Add-Delete',
                         '- `E`: Exchange',
                         '- `zonotope` for projection correlation kernel only)',
                         'Given: {}'.format(self.sampling_mode)]
            raise ValueError('\n'.join(err_print))

        self.list_of_samples.append(chain)

    def sample_mcmc_k_dpp(self, size, **params):
        """ Calls :py:meth:`~sample_mcmc` with ``mode='E'`` and ``params['size'] = size``

        .. seealso::

            - :ref:`finite_dpps_mcmc_sampling`
            - :py:meth:`~FiniteDPP.sample_mcmc`
            - :py:meth:`~FiniteDPP.sample_exact_k_dpp`
            - :py:meth:`~FiniteDPP.flush_samples`
        """

        self.sampling_mode = 'E'

        self.size_k_dpp = size
        params['size'] = size

        self.sample_mcmc(self.sampling_mode, **params)

    def compute_K(self, msg=False):
        """ Compute the correlation kernel :math:`\\mathbf{K}` from the original parametrization of the :class:`FiniteDPP` object.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """

        if self.K is not None:
            # msg = 'K (correlation) kernel available'
            # print(msg)
            pass

        else:
            if not msg:
                print('K (correlation) kernel computed via:')

            if self.K_eig_vals is not None:
                msg = '- U diag(eig_K) U.T'
                print(msg)
                self.K = (self.eig_vecs * self.K_eig_vals).dot(self.eig_vecs.T)

            elif self.A_zono is not None:
                msg = '\n'.join(['- K = A.T (AA.T)^-1 A, using',
                                 '- U = QR(A.T)',
                                 '- K = U U.T'])
                print(msg)
                self.K_eig_vals = np.ones(self.A_zono.shape[0])
                self.eig_vecs, _ = la.qr(self.A_zono.T, mode='economic')
                self.K = self.eig_vecs.dot(self.eig_vecs.T)

            elif self.L_eig_vals is not None:
                msg = '- eig_K = eig_L/(1+eig_L)'
                print(msg)
                self.K_eig_vals = self.L_eig_vals / (1.0 + self.L_eig_vals)
                self.compute_K(msg=True)

            elif self.L is not None:
                msg = '- eigendecomposition of L'
                print(msg)
                self.L_eig_vals, self.eig_vecs = la.eigh(self.L)
                self.compute_K(msg=True)

            else:
                self.compute_L(msg=True)
                self.compute_K(msg=True)

    def compute_L(self, msg=False):
        """ Compute the likelihood kernel :math:`\\mathbf{L}` from the original parametrization of the :class:`FiniteDPP` object.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """

        if self.L is not None:
            # msg = 'L (likelihood) kernel available'
            # print(msg)
            pass

        elif (self.kernel_type == 'correlation') and self.projection:
            err_print = ['L = K(I-K)^-1 = kernel cannot be computed:',
                         'K is projection kernel: some eigenvalues equal 1']
            raise ValueError('\n'.join(err_print))

        else:
            if not msg:
                print('L (likelihood) kernel computed via:')

            if self.L_eig_vals is not None:
                msg = '- U diag(eig_L) U.T'
                print(msg)
                self.L = (self.eig_vecs * self.L_eig_vals).dot(self.eig_vecs.T)

            elif self.L_gram_factor is not None:
                msg = '- L = Phi.T Phi, where Phi = L_gram_factor'
                print(msg)
                self.L = self.L_gram_factor.T.dot(self.L_gram_factor)

            elif self.K_eig_vals is not None:
                try:  # to compute eigenvalues of kernel L = K(I-K)^-1
                    msg = '- eig_L = eig_K/(1-eig_K)'
                    print(msg)
                    np.seterr(divide='raise')
                    self.L_eig_vals = self.K_eig_vals / (1.0 - self.K_eig_vals)
                    self.compute_L(msg=True)
                except FloatingPointError:
                    err_print = ['Eigenvalues of L kernel cannot be computed',
                                 'eig_L = eig_K/(1-eig_K)',
                                 'K kernel has some eig_K very close to 1.',
                                 'Hint: `K` kernel might be a projection.']
                    raise FloatingPointError('\n'.join(err_print))

            elif self.K is not None:
                msg = '- eigendecomposition of K'
                print(msg)
                self.K_eig_vals, self.eig_vecs = la.eigh(self.K)
                self.compute_L(msg=True)

            else:
                self.compute_K(msg=True)
                self.compute_L(msg=True)

    def plot_kernel(self, title=''):
        """Display a heatmap of the kernel used to define the :class:`FiniteDPP` object (correlation kernel :math:`\\mathbf{K}` or likelihood kernel :math:`\\mathbf{L}`)

        :param title:
            Plot title

        :type title:
            string
        """

        fig, ax = plt.subplots(1, 1)

        if self.kernel_type == 'correlation':
            self.compute_K()
            nb_items, kernel_to_plot = self.K.shape[0], self.K
            str_title = r'$K$ (correlation) kernel'

        elif self.kernel_type == 'likelihood':
            self.compute_L()
            nb_items, kernel_to_plot = self.L.shape[0], self.L
            str_title = r'$L$ (likelihood) kernel'

        heatmap = ax.pcolor(kernel_to_plot, cmap='jet')

        ax.set_aspect('equal')

        ticks = np.arange(nb_items)
        ticks_label = [r'${}$'.format(tic) for tic in ticks]

        ax.xaxis.tick_top()
        ax.set_xticks(ticks + 0.5, minor=False)

        ax.invert_yaxis()
        ax.set_yticks(ticks + 0.5, minor=False)

        ax.set_xticklabels(ticks_label, minor=False)
        ax.set_yticklabels(ticks_label, minor=False)

        plt.title(title if title else str_title, y=1.1)

        plt.colorbar(heatmap)
