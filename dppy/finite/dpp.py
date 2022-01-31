# coding: utf8
""" Implementation of :py:class:`FiniteDPP` object which has 6 main methods:

- :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_exact`, see also :ref:`sampling DPPs exactly<finite_dpps_exact_sampling>`
- :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_exact_k_dpp`, see also :ref:`sampling k-DPPs exactly<finite_dpps_exact_sampling>`
- :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc`, see also :ref:`sampling DPPs with MCMC<finite_dpps_mcmc_sampling>`
- :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc_k_dpp`, see also :ref:`sampling k-DPPs with MCMC<finite_dpps_mcmc_sampling_k_dpps>`
- :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_correlation_kernel`, to compute the correlation :math:`K` kernel from initial parametrization
- :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_likelihood_kernel`, to compute the likelihood :math:`L` kernel from initial parametrization

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/index.html>`_
"""


import matplotlib.pyplot as plt
import numpy as np

from dppy.finite.compute_kernels import (
    compute_correlation_kernel,
    compute_likelihood_kernel,
)
from dppy.finite.exact_samplers.select_samplers import (
    select_sampler_exact_dpp,
    select_sampler_exact_k_dpp,
)
from dppy.finite.mcmc_samplers.select_samplers import (
    select_sampler_mcmc_dpp,
    select_sampler_mcmc_k_dpp,
)
from dppy.finite.utils import check_arguments_coherence, check_parameters_validity
from dppy.utils import check_random_state


class FiniteDPP:
    r"""Finite DPP object parametrized by

    :param string kernel_type:
        Indicate if the associated :math:`\operatorname{DPP}` is defined via its

        - ``"correlation"`` :math:`\mathbf{K}` kernel, or
        - ``"likelihood"`` :math:`\mathbf{L}` kernel.

    :param projection:
        Indicate if the associated kernel is of projection type, i.e., :math:`M^2 = M`.
    :type projection:
        bool, default ``False``

    :param hermitian:
        Indicate if the associated kernel is hermitian, i.e., :math:`M^* = M`
    :type hermitian:
        bool, default ``True``

    :param params:
        Dictionary containing the parametrization of the underlying

        - correlation kernel

            - ``"K": K``, with :math:`\mathbf{K} (N \times N)`. If ``hermitian=True`` then :math:`0 \preceq \mathbf{K} \preceq I` must be satisfied,
            - ``"K_eig_dec": (eig_vals, eig_vecs)``, with :math:`0 \leq eig_vals \leq 1` and columns of eig_vecs must be orthonormal,
            - ``"A_zono": A``, with :math:`A (d \times N)` and :math:`\operatorname{rank}(A)=d`.

        - likelihood kernel

            - ``"L": L``, with ``\mathbf{L}`` :math:`\succeq 0`
            - ``"L_eig_dec": (eig_vals, eig_vecs)``, with ``eig_vals`` :math:`\geq 0`,
            - ``"L_gram_factor": Phi``, with :math:`\mathbf{L} = \Phi^{\top} \Phi`,
            - ``"L_eval_X_data": (eval_L, X_data)``, with :math:`X (N \times d)` and ``eval_L`` a likelihood function such that :math:`\mathbf{L} =` ``eval_L` :math:`(X, X)``.

            For a full description of the requirements imposed on ``eval_L``"s interface, see the documentation :func:`dppy.finite.exact_samplers.vfx_samplers.vfx_sampling_precompute_constants`.
            For an example, see the implementation of any of the kernels provided by scikit-learn (e.g. sklearn.gaussian_process.kernels.PairwiseKernel).

    :type params:
        dict

    .. seealso::

        - :ref:`finite_dpps_definition`
        - :ref:`finite_dpps_exact_sampling`
    """

    def __init__(self, kernel_type, projection=False, hermitian=True, **params):

        check_arguments_coherence(kernel_type, projection, hermitian, **params)

        self.kernel_type = kernel_type
        self.hermitian = hermitian
        self.projection = projection

        self.L = params.get("L", None)
        self.K = params.get("K", None)

        self.K_eig_vals, eig_vecs = params.get("K_eig_dec", (None, None))
        self.L_eig_vals, eig_vecs = params.get("L_eig_dec", (None, eig_vecs))
        self.eig_vecs = eig_vecs

        self.A_zono = params.get("A_zono", None)
        self.L_gram_factor = params.get("L_gram_factor", None)
        self.eval_L, self.X_data = params.get("L_eval_X_data", [None, None])

        check_parameters_validity(self)

        # Sampling
        self.list_of_samples = []
        ## k-dpp
        self.size_k_dpp = 0
        self.esp = None  # evaluation of the elementary symmetric polynomials
        ## vfx and alpha samplers
        self.intermediate_sample_info = None

    def flush_samples(self):
        """Empty the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute.

        .. see also::

            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_exact`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc`
        """
        self.list_of_samples = []
        self.size_k_dpp = 0

    def sample_exact(self, method="spectral", random_state=None, **params):
        """Sample exactly from the corresponding :py:class:`~dppy.finite.dpp.FiniteDPP` object. Default sampling method="spectral" assumes the corresponding DPP is  hermitian.

        :param method:
            - ``"spectral"`` (default), see :ref:`finite_dpps_exact_sampling_spectral_method` and :py:func:`~dppy.finite.exact_samplers.spectral_sampler_dpp.spectral_sampler`. It applies to hermitian DPPs: ``FiniteDPP(..., hermitian=True, ...)``.
            - ``"vfx"`` dpp-vfx rejection sampler of :cite:`DeCaVa19`, see :ref:`finite_dpps_exact_sampling_intermediate_sampling_methods` and :py:func:`~dppy.finite.exact_samplers.vfx_samplers.vfx_sampler_dpp`. It applies to ``FiniteDPP("likelihood", hermitian=True, L_eval_X_data=(eval_L, X_data))``,
            - ``"alpha"`` alpha-dpp rejection sampler :cite:`CaDeVa20`, see :ref:`finite_dpps_exact_sampling_intermediate_sampling_methods`. It applies to ``FiniteDPP("likelihood", hermitian=True, L_eval_X_data=(eval_L, X_data))``.
            - ``"Schur"``, conditionals are computed as Schur complements see :eq:`eq:chain_rule_schur`. It applies to ``FiniteDPP("correlation", projection=True, ...)``.
            - ``"Chol"``, see :ref:`finite_dpps_exact_sampling_generic_method`. It refers to :cite:`Pou19` Algorithm 1 (see ``"generic"``) and Algorithm 3 which applies to``FiniteDPP("correlation", projection=True, hermitian=true, ...)``.
            - ``"generic"``, see :ref:`finite_dpps_exact_sampling_generic_method`, applies to generic (hermitian or not) finite DPPs, :cite:`Pou19` Algorithm 1.

        :type method:
            string, default ``"spectral"``

        :param dict params:
            Dictionary containing the parameters of the corresponding exact sampling method

            - ``method="spectral"``
                - ``"mode"``
                    - ``"GS"`` (default): similar to Algorithm 2 of :cite:`Gil14`  and Algorithm 3 of :cite:`TrBaAm18`.
                    - ``"GS_bis"``: slight modification of ``"GS"``
                    - ``"KuTa12"``: corresponds to Algorithm 1 of :cite:`KuTa12`

            - ``method="vfx"``

                See :py:meth:`~dppy.finite.exact_samplers.vfx_samplers.vfx_sampler_dpp` for a full list of all parameters accepted by "vfx" sampling. We report here the most impactful

                + ``"rls_oversample_dppvfx"`` (default 4.0) Oversampling parameter used to construct dppvfx's internal Nystrom approximation. This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections.
                + ``"rls_oversample_bless"`` (default 4.0) Oversampling parameter used during bless's internal Nystrom approximation. This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections

                Empirically, a small factor [2,10] seems to work for both parameters. It is suggested to start s.t. a small number and increase if the algorithm fails to terminate.

            - ``method="alpha"``

                See :py:meth:`~dppy.finite.exact_samplers.alpha_samplers.alpha_sampler_k_dpp` for a full list of all parameters accepted by "alpha" sampling. We report here the most impactful

                + ``"rls_oversample_alphadpp"`` (default 4.0) Oversampling parameter used to construct alpha-dpp's internal Nystrom approximation. This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections.
                + ``"rls_oversample_bless"`` (default 4.0) Oversampling parameter used during bless's internal Nystrom approximation. This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections

                Empirically, a small factor [2,10] seems to work for both parameters. It is suggested to start with
                a small number and increase if the algorithm fails to terminate.

        :return:
            Returns a sample from the corresponding :py:class:`~dppy.finite.dpp.FiniteDPP` object. In any case, the sample is appended to the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute as a list.
        :rtype:
            list

        .. note::

            Each time you call this method, the sample is appended to the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute as a list.

            The :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute can be emptied using :py:meth:`~dppy.finite.dpp.FiniteDPP.flush_samples`

        .. seealso::

            - :ref:`finite_dpps_exact_sampling`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.flush_samples`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc`
        """
        rng = check_random_state(random_state)
        sampler = select_sampler_exact_dpp(self, method)
        sample = sampler(self, rng, **params)

        self.list_of_samples.append(sample)
        return sample

    def sample_exact_k_dpp(self, size, method="spectral", random_state=None, **params):
        r"""Sample exactly from :math:`\operatorname{k-DPP}`. A priori the :py:class:`~dppy.finite.dpp.FiniteDPP` object was instanciated by its likelihood :math:`\mathbf{L}` kernel so that

        .. math::

            \mathbb{P}_{\operatorname{k-DPP}}(\mathcal{X} = S)
                \propto \det \mathbf{L}_S ~ 1_{|S|=k}

        :param size:
            size :math:`k` of the :math:`\operatorname{k-DPP}`

        :type size:
            int

        :param mode:
            - ``projection=True``:
                - ``"GS"`` (default): Gram-Schmidt on the rows of :math:`\mathbf{K}`.
                - ``"Schur"``: Use Schur complement to compute conditionals.

            - ``projection=False``:
                - ``"GS"`` (default): Gram-Schmidt on the rows of the eigenvectors of :math:`\mathbf{K}` selected in Phase 1.
                - ``"GS_bis"``: Slight modification of ``"GS"``
                - ``"KuTa12"``: Algorithm 1 in :cite:`KuTa12`
                - ``"vfx"``: the dpp-vfx rejection sampler in :cite:`DeCaVa19`
                - ``"alpha"``: the alpha-dpp rejection sampler in :cite:`CaDeVa20`

        :type mode:
            string, default ``"GS"``

        :param dict params:
            Dictionary containing the parameters for exact samplers with keys

            - If ``mode="vfx"``

                See :py:meth:`~dppy.finite.exact_samplers.vfx_sampler_k_dpp` for a full list of all parameters accepted by "vfx" sampling. We report here the most impactful

                + ``"rls_oversample_dppvfx"`` (default 4.0) Oversampling parameter used to construct dppvfx's internal Nystrom approximation. This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections.
                + ``"rls_oversample_bless"`` (default 4.0) Oversampling parameter used during bless's internal Nystrom approximation. This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections

                Empirically, a small factor [2,10] seems to work for both parameters. It is suggested to start with
                a small number and increase if the algorithm fails to terminate.

            - If ``mode="alpha"``
                See :py:meth:`~dppy.finite.exact_samplers.alpha_samplers.alpha_sampler_k_dpp` for a full list of all parameters accepted by "alpha" sampling. We report here the most impactful

                + ``"rls_oversample_alphadpp"`` (default 4.0) Oversampling parameter used to construct alpha-dpp's internal Nystrom approximation. This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections.
                + ``"rls_oversample_bless"`` (default 4.0) Oversampling parameter used during bless's internal Nystrom approximation. This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections
                + ``"early_stop"`` (default False) Wheter to return as soon as a k-DPP sample is drawn, or to continue with alpha-dpp internal binary search to make subsequent sampling faster.

                Empirically, a small factor [2,10] seems to work for both parameters. It is suggested to start with
                a small number and increase if the algorithm fails to terminate.

        :return:
            A sample from the corresponding :math:`\operatorname{k-DPP}`.

            In any case, the sample is appended to the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute as a list.

        :rtype:
            list

        .. note::

            Each time you call this method, the sample is appended to the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute as a list.

            The :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute can be emptied using :py:meth:`~dppy.finite.dpp.FiniteDPP.flush_samples`

        .. caution::

            The underlying kernel :math:`\mathbf{K}`, resp. :math:`\mathbf{L}` must be real valued for now.

        .. seealso::

            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_exact`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc_k_dpp`
        """
        rng = check_random_state(random_state)
        sampler = select_sampler_exact_k_dpp(self, method)
        sample = sampler(self, size, rng, **params)

        self.size_k_dpp = size
        self.list_of_samples.append(sample)
        return sample

    def sample_mcmc(self, method="aed", random_state=None, **params):
        """Run a MCMC with stationary distribution the corresponding :py:class:`~dppy.finite.dpp.FiniteDPP` object.

        :param string method:

            - ``"aed"`` add-exchange-delete
            - ``"ad"`` add-delete
            - ``"e"`` exchange
            - ``"zonotope"`` Zonotope sampling

        :param dict params:
            Dictionary containing the parameters for MCMC samplers with keys

            - If ``method="aed","ad","e"``

                + ``"s_init"`` (default None) Starting state of the Markov chain
                + ``"nb_iter"`` (default 10) Number of iterations of the chain
                + ``"T_max"`` (default None) Time horizon
                + ``"size"`` (default None) Size of the initial sample for ``method="AD"/"E"``

                    * :math:`\\operatorname{rank}(\\mathbf{K})=\\operatorname{trace}(\\mathbf{K})` for projection :math:`\\mathbf{K}` (correlation) kernel and ``method="E"``

            - If ``method="zonotope"``:

                + ``"lin_obj"`` linear objective in main optimization problem (default np.random.randn(N))
                + ``"x_0"`` initial point in zonotope (default A*u, u~U[0,1]^n)
                + ``"nb_iter"`` (default 10) Number of iterations of the chain
                + ``"T_max"`` (default None) Time horizon

        :return:
            The last sample of the trajectory of Markov chain.

            In any case, the full trajectory of the Markov chain, made of ``params["nb_iter"]`` samples, is appended to the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute as a list of lists.

        :rtype:
            list

        .. note::

            Each time you call this method, the full trajectory of the Markov chain, made of ``params["nb_iter"]`` samples, is appended to the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute as a list of lists.

            The :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute can be emptied using :py:meth:`~dppy.finite.dpp.FiniteDPP.flush_samples`

        .. seealso::

            - :ref:`finite_dpps_mcmc_sampling`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_exact`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.flush_samples`
        """
        rng = check_random_state(random_state)
        sampler = select_sampler_mcmc_dpp(self, method)
        chain = sampler(self, rng, **params)

        self.list_of_samples.append(chain)
        return chain[-1]

    def sample_mcmc_k_dpp(self, size, method="e", random_state=None, **params):
        """Equivalent to :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc` with ``method="E"`` and ``params["size"] = size``

        .. seealso::

            - :ref:`finite_dpps_mcmc_sampling`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_exact_k_dpp`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.flush_samples`
        """
        self.size_k_dpp = size
        sampler = select_sampler_mcmc_k_dpp(self, method)
        chain = sampler(self, size=size, random_state=None, **params)

        self.list_of_samples.append(chain)
        return chain[-1]

    def compute_K(self):
        """Alias of :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_correlation_kernel`."""
        return self.compute_correlation_kernel()

    def compute_correlation_kernel(self):
        r"""Compute the correlation kernel :math:`\mathbf{K}` from the current parametrization of the :py:class:`~dppy.finite.dpp.FiniteDPP` object.

        The returned kernel is also stored as the :py:attr:`~dppy.finite.dpp.FiniteDPP.K` attribute.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """
        return compute_correlation_kernel(self)

    def compute_L(self):
        """Alias of :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_likelihood_kernel`."""
        return self.compute_likelihood_kernel()

    def compute_likelihood_kernel(self):
        r"""Compute the likelihood kernel :math:`\mathbf{L}` from the current parametrization of the :py:class:`~dppy.finite.dpp.FiniteDPP` object.

        The returned kernel is also stored as the :py:attr:`~dppy.finite.dpp.FiniteDPP.L` attribute.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """
        return compute_likelihood_kernel(self)

    def plot_kernel(self, kernel_type="correlation", ax=None):
        r"""Display a heatmap of the kernel used to define the :py:class:`~dppy.finite.dpp.FiniteDPP` object (correlation kernel :math:`\mathbf{K}` or likelihood kernel :math:`\mathbf{L}`)

        :param str kernel_type: ``"correlation"`` or ``"likelihood"``. Default to ``"correlation"``.
        """
        if not kernel_type:
            kernel_type = self.kernel_type

        if kernel_type == "correlation":
            kernel = self.compute_correlation_kernel()
        elif kernel_type == "likelihood":
            kernel = self.compute_likelihood_kernel()
        else:
            raise ValueError(
                "kernel_type must be either 'correlation' or 'likelihood'."
            )

        if ax is None:
            fig, ax = plt.subplots(1, 1)

        heatmap = ax.pcolor(kernel, cmap="jet", vmin=-0.3, vmax=1)

        ticks = np.arange(len(kernel))
        ticks_label = [r"${}$".format(tic) for tic in ticks]

        ax.xaxis.tick_top()
        ax.set_xticks(ticks + 0.5, minor=False)

        ax.invert_yaxis()
        ax.set_yticks(ticks + 0.5, minor=False)

        ax.set_xticklabels(ticks_label, minor=False)
        ax.set_yticklabels(ticks_label, minor=False)

        ax.set_aspect("equal")

        plt.colorbar(heatmap)

        return ax
