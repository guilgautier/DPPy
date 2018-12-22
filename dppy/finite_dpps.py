# coding: utf-8

from dppy.exact_sampling import *
from dppy.mcmc_sampling import *

import matplotlib.pyplot as plt
from warnings import warn


class FiniteDPP:
    """ Finite DPP object parametrized by

    :param kernel_type:

        - ``'inclusion'`` :math:`\mathbf{K}` kernel
        - ``'marginal'`` :math:`\mathbf{L}` kernel

    :type kernel_type:
        string

    :param projection:
        Indicate whether the provided kernel is of projection type. This may be useful when the :class:`FiniteDPP` object is defined through its inclusion kernel :math:`\mathbf{K}`.

    :type projection:
        bool, default ``False``

    :param params:
        Dictionary containing the parametrization of the underlying

        - inclusion kernel

            - ``{'K': K}``, with :math:`0 \preceq \mathbf{K} \preceq I`
            - ``{'K_eig_dec': (eig_vals, eig_vecs)}``, with :math:`0 \leq eigvals \leq 1`
            - ``{'A_zono': A}``, with :math:`A (d \\times N)` and :math:`\operatorname{rank}(A)=d`

        - marginal kernel

            - ``{'L': L}``, with :math:`\mathbf{L}\succeq 0`
            - ``{'L_eig_dec': (eig_vals, eig_vecs)}``, with :math:`eigvals \geq 0`
            - ``{'L_gram_factor': Phi}``, with :math:`\mathbf{L} = \Phi^{ \\top} \Phi`

    :type params:
        dict

    .. caution::

        For now we only consider real valued matrices :math:`\mathbf{K}, \mathbf{L}, A, \Phi`.

    .. seealso::

        - :ref:`finite_dpps_definition`
        - :ref:`finite_dpps_exact_sampling_projection_dpps`
    """

    ###############
    # Constructor #
    ###############
    def __init__(self, kernel_type, projection=False, **params):

        self.kernel_type = kernel_type
        self.__check_kernel_type_arg()

        self.projection = projection
        self.__check_projection_arg()

        # Parameters of the DPP
        self.params_keys = set(params.keys())

        # Inclusion kernel K: P(S C X) = det(K_S)
        self.K = params.get('K', None)
        # If eigendecomposition available: K_eig_dec = [eig_vals, eig_vecs]
        self.K_eig_vals, self.eig_vecs = params.get('K_eig_dec', [None, None])
        # In case projection DPP defined by K_eig_dec, Phase 1 of exact sampling can be bypassed
        self.__proj_eig_vals_1 = None
        # see __check_eig_vals_equal_O1 in __chech_params_validity

        # If full row rank feature matrix passed via 'A_zono' it means that there is the underlying projection kernel is K = A.T (AA.T)^-1 A. A priori, you want to use zonotope approximate sampler.
        if 'A_zono' in self.params_keys:
            self.A_zono = params.get('A_zono')

        # Marginal kernel L: P(X=S) propto det(L_S) = det(L_S)/det(I+L)
        self.L = params.get('L', None)
        # If eigendecomposition available: L_eig_dec = [eig_vals, eig_vecs]
        self.L_eig_vals, self.eig_vecs =\
            params.get('L_eig_dec',
                       [None, None if self.eig_vecs is None
                        else self.eig_vecs])
        # If L defined as Gram matrix: L = Phi.T Phi, with feat matrix Phi dxN
        if 'L_gram_factor' in self.params_keys:
            self.L_gram_factor = params.get('L_gram_factor', None)
            # In case d<N, use 'dual' view
            self.L_dual = None  # L' = Phi Phi.T
            self.L_dual_eig_vals, self.L_dual_eig_vecs = None, None

        self.__check_params_validity()

        # Sampling
        self.mode = None
        # Exact:
        # if K (inclusion) kernel is projection
        # - ``'GS'`` for Gram-Schmidt
        # else
        # - ``'GS'``
        # - ``'GS_bis'`` slight modif of Gram-Schmidt
        # - ``'KuTa12'`` for Kulesza (Algo 1).
        #
        # Approximate:
        # Local chains
        # - 'AED' Add-Exchange-Delete
        # - 'AD' Add-Delete
        # - 'E' Exchange
        # Zonotope
        # No argument to be passed, implicit if A_zono given
        self.list_of_samples = []

    def __str__(self):
        str_info = ['DPP defined through {} {} kernel'.format(
                        'projection' if self.projection else '',
                        self.kernel_type),
                    'Parametrized by {}'.format(self.params_keys),
                    '- sampling mode = {}'.format(self.mode),
                    '- number of samples = {}'.format(
                        len(self.list_of_samples))]

        return '\n'.join(str_info)

    #########################
    # Hidden object methods #
    #########################

    # Check routines
    def __check_kernel_type_arg(self):
        # Ensemble type
        if self.kernel_type not in ['inclusion', 'marginal']:
            err_print = ['Invalid `kernel_type` argument, choose among:',
                         '- `inclusion`: inclusion kernel, P(SCX) = det(K_S)',
                         '- `marginal`: marginal kernel, P(X=S) prto det(L_S)']
            raise ValueError('\n'.join(err_print))

    def __check_projection_arg(self):
        if not isinstance(self.projection, bool):
            err_print = 'Invalid `projection` argument: must be True/False'
            raise ValueError(err_print)

    def __check_params_validity(self):

        # Check initialization parameters of the DPP

        # For inclusion kernel
        if self.kernel_type == 'inclusion':

            auth_params = {'K', 'K_eig_dec', 'A_zono'}
            if auth_params.intersection(self.params_keys):

                if self.K is not None:
                    self.__check_symmetry_of_kernel(self.K)

                    if self.projection:
                        self.__check_is_projection_kernel(self.K)

                elif self.K_eig_vals is not None:

                    if self.projection:
                        self.__check_eig_vals_equal_O1(self.K_eig_vals)
                    else:
                        self.__check_eig_vals_in_01(self.K_eig_vals)

                elif self.A_zono is not None:
                    # A_zono (dxN) must be full row rank, sanity check is d<=N
                    self.__check_is_full_row_rank(self.A_zono)
                    if not self.projection:
                        warn('Weird setting: inclusion kernel defined via `A_zono` but `projection`=False. `projection` switched to True')
                        self.projection = True

            else:
                err_print = ['Invalid inclusion kernel, choose among:',
                             '- `K`: 0 <= K <= I',
                             '- `K_eig_dec`: (eig_vals, eig_vecs) 0 <= eig_vals <= 1',
                             '- `A_zono`: A is dxN matrix, with rank(A)=d corresponding to K = A.T (AA.T)^-1 A',
                             'Given: {}'.format(self.params_keys)]
                raise ValueError('\n'.join(err_print))

        # For marginal kernel
        elif self.kernel_type == 'marginal':

            auth_params = {'L', 'L_eig_dec', 'L_gram_factor'}
            if auth_params.intersection(self.params_keys):

                if self.L is not None:
                    self.__check_symmetry_of_kernel(self.L)

                    if self.projection:
                        self.__check_is_projection_kernel(self.L)

                elif self.L_eig_vals is not None:
                        self.__check_eig_vals_geq_0(self.L_eig_vals)
                        # self.__check_eig_vals_equal_O1(self.L_eig_vals)

                elif self.L_gram_factor is not None:
                    self.__check_compute_L_dual_or_not(self.L_gram_factor)

                    if self.projection:
                        warn('`L_gram_factor`+`projection`=True is a very weird setting, may switch to `projection`=False')

            else:
                err_print = ['Invalid marginal kernel, choose among:',
                             '- `L`: L >= 0',
                             '- `L_eig_dec`: (eig_vals, eig_vecs)',
                             '- `L_gram_factor`: Phi is dxN feature matrix corresponding to L = Phi.T Phi',
                             'Given: {}'.format(self.params_keys)]
                raise ValueError('\n'.join(err_print))

    def __check_symmetry_of_kernel(self, kernel):

        if not np.allclose(kernel.T, kernel):
            err_print = 'Invalid kernel: not symmetric'
            raise ValueError(err_print)

    def __check_is_projection_kernel(self, kernel):
            # Cheap test to check reproducing property
            nb_to_check = np.min([5, kernel.shape[0]])
            items_to_check = np.arange(nb_to_check)
            K_i_ = kernel[items_to_check, :]
            K_ii = kernel[items_to_check, items_to_check]

            if not np.allclose(np_inner1d(K_i_, K_i_), K_ii):
                raise ValueError('Invalid kernel: not projection')
            else:
                pass

    def __check_eig_vecs_orthonormal(self, eig_vecs):
        # Cheap test for checking orthonormality of eigenvectors

        nb_to_check = np.min([5, eig_vecs.shape[0]])
        V_j = eig_vecs[:, :nb_to_check]

        if not np.allclose(V_j.T.dot(V_j), np.eye(nb_to_check)):
            raise ValueError('Invalid eigenvectors: not orthonormal')
        else:
            pass

    def __check_eig_vals_equal_O1(self, eig_vals):

        tol = 1e-8
        eig_vals_0 = np.abs(eig_vals) <= tol
        eig_vals_1 = np.abs(1 - eig_vals) <= tol
        eig_vals_0_or_1 = eig_vals_0 ^ eig_vals_1  # ^ = xor

        if not np.all(eig_vals_0_or_1):
            raise ValueError('Invalid kernel: does not seem to be a projection, check that the eigenvalues provided are equal to 0 or 1')
        else:
            # Record eigenvectors that have eigenvalues = 1 for later use in sampling phase
            self.__proj_eig_vals_1 = np.where(eig_vals_1)[0]

    def __check_eig_vals_in_01(self, eig_vals):

        tol = 1e-8
        if not np.all((-tol <= eig_vals) & (eig_vals <= 1.0 + tol)):
            err_print = 'Invalid inclusion kernel, eigenvalues not in [0,1]'
            raise ValueError(err_print)
        else:
            pass

    def __check_eig_vals_geq_0(self, eig_vals):

        tol = 1e-8
        if not np.all(eig_vals >= -tol):
            err_print = 'Invalid marginal kernel, eigenvalues not >= 0'
            raise ValueError(err_print)
        else:
            pass

    def __check_is_full_row_rank(self, A_zono):

        d, N = A_zono.shape

        if d > N:
            err_print = ['Invalid `A_zono` (dxN) parameter,\
                         not full row rank: d(={}) > N(={})'.format(d, N)]
            raise ValueError(err_print)

        else:
            rank = np.linalg.matrix_rank(A_zono)

            if rank != d:
                err_print = ['Invalid `A_zono` (dxN) parameter,\
                             not full row rank: d(={}) != rank(={})'.format(d, rank)]
                raise ValueError(err_print)

            else:
                pass

    def __check_compute_L_dual_or_not(self, L_gram_factor):

        d, N = L_gram_factor.shape

        if d < N:
            self.L_dual = L_gram_factor.dot(L_gram_factor.T)
            str_print = 'd={} < N={}: L dual kernel was computed'.format(d, N)

        else:
            self.L = L_gram_factor.T.dot(L_gram_factor)
            str_print = 'd={} >= N={}: L kernel was computed'.format(d, N)

        print(str_print)

    ##################
    # Object methods #
    ##################
    def info(self):
        """ Display infos about the :class:`FiniteDPP` object
        """
        print(self.__str__())

    def flush_samples(self):
        """ Empty the ``FiniteDPP.list_of_samples`` attribute.

        .. see also::

            - :func:`sample_exact <sample_exact>`
            - :func:`sample_mcmc <sample_mcmc>`
        """
        self.list_of_samples = []

    # Exact sampling
    def sample_exact(self, mode='GS'):
        """ Sample exactly from the corresponding :class:`FiniteDPP <FiniteDPP>` object. The sampling scheme is based on the chain rule with Gram-Schmidt like updates of the conditionals.

        :param mode:

            - ``projection=True``:
                - ``'GS'`` (default): Gram-Schmidt on the rows of :math:`\mathbf{K}`.
                - ``'Schur'``: Use Schur complement to compute conditionals.

            - ``projection=False``:
                - ``'GS'`` (default): Gram-Schmidt on the rows of the eigenvectors of :math:`\mathbf{K}` selected in Phase 1.
                - ``'GS_bis'``: Slight modification of ``'GS'``
                - ``'KuTa12'``: Algorithm 1 in :cite:`KuTa12`
        :type mode:
            string, default ``'GS'``

        :return:
            A sample from the corresponding :class:`FiniteDPP <FiniteDPP>` object.
        :rtype:
            list

        .. note::

            Each time you call this function, the sample is added to the ``FiniteDPP.list_of_samples`` attribute.

            The latter can be emptied using :func:`.flush_samples() <flush_samples>`

        .. caution::

            The underlying kernel :math:`\mathbf{K}`, resp. :math:`\mathbf{L}` must be real valued for now.

        .. seealso::

            - :ref:`finite_dpps_exact_sampling`
            - :func:`flush_samples <flush_samples>`
            - :func:`sample_mcmc <sample_mcmc>`
        """

        self.mode = mode

        # If eigen decoposition of K, L or L_dual is available USE IT!

        # If projection DPP defined by K_eig_dec
        if self.__proj_eig_vals_1 is not None:
            # Phase 1 is bypassed
            V = self.eig_vecs[:, self.__proj_eig_vals_1]
            # Phase 2
            sampl = proj_dpp_sampler_eig(V, self.mode)
            self.list_of_samples.append(sampl)

        elif self.K_eig_vals is not None:
            # Phase 1
            V = dpp_eig_vecs_selector(self.K_eig_vals, self.eig_vecs)
            # Phase 2
            sampl = proj_dpp_sampler_eig(V, self.mode)
            self.list_of_samples.append(sampl)

        elif self.L_eig_vals is not None:
            self.K_eig_vals = self.L_eig_vals / (1.0 + self.L_eig_vals)
            self.__check_eig_vals_in_01(self.K_eig_vals)
            self.sample_exact()

        elif 'L_gram_factor' in self.params_keys:
        # If DPP is marginal kernel with parameter 'L_gram_factor' i.e. L = Phi.T Phi but dual kernel L' = Phi Phi.T was cheaper to use (computation of L' and diagonalization for sampling)
            if self.L_dual_eig_vals is not None:
                # Phase 1
                V = dpp_eig_vecs_selector_L_dual(self.L_dual_eig_vals,
                                                 self.L_dual_eig_vecs,
                                                 self.L_gram_factor)
                # Phase 2
                sampl = proj_dpp_sampler_eig(V, self.mode)
                self.list_of_samples.append(sampl)

            elif self.L_dual is not None:
                self.L_dual_eig_vals, self.L_dual_eig_vecs =\
                    la.eigh(self.L_dual)
                self.__check_eig_vals_geq_0(self.L_dual_eig_vals)
                self.sample_exact()

        # Otherwise
        # If DPP defined through inclusion kernel with projection kernel no need of eigendecomposition, you can apply Gram-Schmidt on the columns of K (equiv rows because of symmetry)
        elif (self.K is not None) and self.projection:
            sampl = proj_dpp_sampler_kernel(self.K, self.mode)
            self.list_of_samples.append(sampl)

        # If DPP defined through inclusion kernel with generic kernel, eigen-decompose it
        elif self.K is not None:
            self.K_eig_vals, self.eig_vecs = la.eigh(self.K)
            self.sample_exact()

        # If DPP defined through marginal kernel with kernel L, eigen-decompose it
        elif self.L is not None:
            self.L_eig_vals, self.eig_vecs = la.eigh(self.L)
            self.sample_exact()

        # If DPP defined through inclusion kernel with parameter 'A_zono', a priori you wish to use the zonotope approximate sampler: warning is raised.
        elif 'A_zono' in self.params_keys:
            warn('DPP defined via `A_zono`, apriori you want to use `sampl_mcmc`, but you have called `sample_exact`')
            self.projection, self.mode = True, 'GS'

            self.eig_vals = np.ones(self.A_zono.shape[0])
            self.eig_vecs, _ = la.qr(self.A_zono.T, mode='economic')

            sampl = proj_dpp_sampler_eig(self.eig_vecs, self.mode)
            self.list_of_samples.append(sampl)

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

            - If ``mode='AED','AD','E'``

                + ``'s_init'`` (default None) Starting state of the Markov chain
                + ``'nb_iter'`` (default 10) Number of iterations of the chain
                + ``'T_max'`` (default None) Time horizon
                + ``'size'`` (default None) Size of the initial sample for ``mode='AD'/'E'``

                    * :math:`\operatorname{rank}(\mathbf{K})=\operatorname{Tr}(\mathbf{K})` for projection :math:`\mathbf{K}` (inclusion) kernel and ``mode='E'``
            - If ``mode='zonotope'``:

        :type params:
            dict

        :return:
            A sample from the corresponding :class:`FiniteDPP <FiniteDPP>` object.
        :rtype:
            list

        .. seealso::

            - :ref:`finite_dpps_mcmc_sampling`
            - :func:`sample_exact <sample_exact>`
            - :func:`flush_samples <flush_samples>`
        """

        auth_sampl_mod = ['AED', 'AD', 'E', 'zonotope']

        if mode in auth_sampl_mod:
            self.mode = mode

            if self.mode == 'zonotope':
                if 'A_zono' in self.params_keys:
                    MC_samples = zonotope_sampler(self.A_zono, **params)

                else:
                    err_print = ['Invalid `mode=zonotope` parameter.',
                                 'DPP must be defined via `A_zono`',
                                 'Given: {}'.format(self.params_keys)]
                    raise ValueError(' '.join(err_print))

            elif self.mode == 'E':
                if (self.kernel_type == 'inclusion') and self.projection:
                    self.compute_K()
                    # |sample|=Tr(K) a.s. for projection DPP(K)
                    params.update({'size': int(np.round(np.trace(self.K)))})

                    MC_samples = dpp_sampler_mcmc(self.K, self.mode, **params)
                else:
                    self.compute_L()
                    MC_samples = dpp_sampler_mcmc(self.L, self.mode, **params)

            elif self.mode in ('AED', 'AD'):
                self.compute_L()
                MC_samples = dpp_sampler_mcmc(self.L, self.mode, **params)

            self.list_of_samples.append(MC_samples)

        else:
            err_print = ['Invalid `mode` parameter, choose among:',
                         '- `AED` for Add-Exchange-Delete',
                         '- `AD` for Add-Delete',
                         '- `E` for Exchange',
                         '- `zonotope` projection inclusion kernel only)',
                         'Given: {}'.format(self.mode)]
            raise ValueError('\n'.join(err_print))

    def compute_K(self, msg=False):
        """ Compute the inclusion kernel :math:`\mathbf{K}` from the original parametrization of the :class:`FiniteDPP` object.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """

        if self.K is not None:
            msg = 'K (inclusion) kernel available'
            print(msg)

        else:
            if msg:
                pass
            else:
                print('K (inclusion) kernel computed via:')

            if self.K_eig_vals is not None:
                msg = '- U diag(eig_K) U.T'
                print(msg)
                self.K = (self.eig_vecs * self.K_eig_vals).dot(self.eig_vecs.T)

            elif 'A_zono' in self.params_keys:
                msg = ('- K = A.T (AA.T)^-1 A, using',
                       '- U = QR(A.T)',
                       '- K = U U.T')
                print('\n'.join(msg))
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
                self.__check_eig_vals_geq_0(self.L_eig_vals)
                self.compute_K(msg=True)

            else:
                self.compute_L(msg=True)
                self.compute_K(msg=True)

    def compute_L(self, msg=False):
        """ Compute the marginal kernel :math:`\mathbf{L}` from the original parametrization of the :class:`FiniteDPP` object.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """

        if self.L is not None:
            msg = 'L (marginal) kernel available'
            print(msg)

        elif (self.kernel_type == 'inclusion') and self.projection:
            err_print = ['L = K(I-K)^-1 = kernel cannot be computed:',
                         'K is projection kernel: some eigenvalues equal 1']
            raise ValueError('\n'.join(err_print))

        else:
            if msg:
                pass
            else:
                print('L (marginal) kernel computed via:')

            if 'L_gram_factor' in self.params_keys:
                msg = '- `L_gram_factor` i.e. L = Phi.T Phi'
                print(msg)
                self.L = self.L_gram_factor.T.dot(self.L_gram_factor)

            elif self.L_eig_vals is not None:
                msg = '- U diag(eig_L) U.T'
                print(msg)
                self.L = (self.eig_vecs * self.L_eig_vals).dot(self.eig_vecs.T)

            elif self.K_eig_vals is not None:
                try:  # to compute eigenvalues of kernel L = K(I-K)^-1
                    msg = '- eig_L = eig_K/(1-eig_K)'
                    print(msg)
                    np.seterr(divide='raise')
                    self.L_eig_vals = self.K_eig_vals / (1.0 - self.K_eig_vals)
                    self.compute_L(msg=True)
                except:
                    err_print = ['Eigenvalues of L kernel cannot be computed',
                                 'eig_L = eig_K/(1-eig_K)',
                                 'K kernel has some eig_K very close to 1.',
                                 'Hint: `K` kernel might be a projection.']
                    raise FloatingPointError('\n'.join(err_print))

            elif self.K is not None:
                msg = '- eigendecomposition of K'
                print(msg)
                self.K_eig_vals, self.eig_vecs = la.eigh(self.K)
                self.__check_eig_vals_in_01(self.K_eig_vals)
                self.compute_L(msg=True)

            else:
                self.compute_K(msg=True)
                self.compute_L(msg=True)

    def plot_kernel(self, title=''):
        """Display a heatmap of the kernel used to define the :class:`FiniteDPP` object (inclusion kernel :math:`\mathbf{K}` or marginal kernel :math:`\mathbf{L}`)

        :param title:
            Plot title

        :type title:
            string
        """

        fig, ax = plt.subplots(1, 1)

        if self.kernel_type == 'inclusion':
            if self.K is None:
                self.compute_K()
            self.nb_items = self.K.shape[0]
            kernel_to_plot = self.K
            str_title = r'$K$ (inclusion) kernel'

        elif self.kernel_type == 'marginal':
            if self.L is None:
                self.compute_L()
            self.nb_items = self.L.shape[0]
            kernel_to_plot = self.L
            str_title = r'$L$ (marginal) kernel'

        heatmap = ax.pcolor(kernel_to_plot, cmap='jet')

        ax.set_aspect('equal')

        ticks = np.arange(self.nb_items)
        ticks_label = [r'${}$'.format(tic) for tic in ticks]

        ax.xaxis.tick_top()
        ax.set_xticks(ticks + 0.5, minor=False)

        ax.invert_yaxis()
        ax.set_yticks(ticks + 0.5, minor=False)

        ax.set_xticklabels(ticks_label, minor=False)
        ax.set_yticklabels(ticks_label, minor=False)

        plt.title(title if title else str_title, y=1.1)

        plt.colorbar(heatmap)
        plt.show()
