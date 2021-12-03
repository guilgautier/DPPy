# coding: utf8
""" Implementation of :py:class:`FiniteDPP` object which has 6 main methods:

- :py:meth:`~FiniteDPP.sample_exact`, see also :ref:`sampling DPPs exactly<finite_dpps_exact_sampling>`
- :py:meth:`~FiniteDPP.sample_exact_k_dpp`, see also :ref:`sampling k-DPPs exactly<finite_dpps_exact_sampling>`
- :py:meth:`~FiniteDPP.sample_mcmc`, see also :ref:`sampling DPPs with MCMC<finite_dpps_mcmc_sampling>`
- :py:meth:`~FiniteDPP.sample_mcmc_k_dpp`, see also :ref:`sampling k-DPPs with MCMC<finite_dpps_mcmc_sampling_k_dpps>`
- :py:meth:`~FiniteDPP.compute_K`, to compute the correlation :math:`K` kernel from initial parametrization
- :py:meth:`~FiniteDPP.compute_L`, to compute the likelihood :math:`L` kernel from initial parametrization

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/index.html>`_
"""


from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as la

# EXACT
from dppy.finite.exact_samplers.alpha_samplers import (
    alpha_sampler_dpp,
    alpha_sampler_k_dpp,
)
from dppy.finite.exact_samplers.chol_sampler import chol_sampler, chol_sampler_k_dpp
from dppy.finite.exact_samplers.generic_samplers import generic_sampler
from dppy.finite.exact_samplers.schur_sampler import schur_sampler, schur_sampler_k_dpp
from dppy.finite.exact_samplers.spectral_sampler_dpp import spectral_sampler
from dppy.finite.exact_samplers.spectral_sampler_k_dpp import spectral_sampler_k_dpp
from dppy.finite.exact_samplers.vfx_samplers import vfx_sampler_dpp, vfx_sampler_k_dpp

# MCMC
from dppy.finite.mcmc_samplers.add_delete_sampler import add_delete_sampler
from dppy.finite.mcmc_samplers.add_exchange_delete_sampler import (
    add_exchange_delete_sampler,
)
from dppy.finite.mcmc_samplers.exchange_sampler import exchange_sampler
from dppy.finite.mcmc_samplers.zonotope_sampler import zonotope_sampler

# UTILS
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
        Indicate if the associated kernel is hermitian, i.e., :math:`M^\dagger = M`
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

        self.params_keys = set(params.keys())

        # Sampling
        self.list_of_samples = []
        ## k-dpp
        self.size_k_dpp = 0
        self.esp = None  # evaluation of the elementary symmetric polynomials
        ## vfx and alpha samplers
        self.intermediate_sample_info = None

    def __str__(self):
        str_info = [
            "DPP defined through {} {} kernel".format(
                "projection" if self.projection else "", self.kernel_type
            ),
            "Parametrized by {}".format(self.params_keys),
        ]

        return "\n".join(str_info)

    # Check routine

    def info(self):
        """Display infos about the :class:`FiniteDPP` object"""
        print(self.__str__())

    def flush_samples(self):
        """Empty the :py:attr:`~FiniteDPP.list_of_samples` attribute.

        .. see also::

            - :py:meth:`~FiniteDPP.sample_exact`
            - :py:meth:`~FiniteDPP.sample_mcmc`
        """
        self.list_of_samples = []
        self.size_k_dpp = 0

    def sample_exact(self, method="spectral", random_state=None, **params):
        """Sample exactly from the corresponding :class:`FiniteDPP <FiniteDPP>` object. Default sampling method="spectral" assumes the corresponding DPP is  hermitian.

        :param method:
            - ``"spectral"`` (default), see :ref:`finite_dpps_exact_sampling_spectral_method`. It applies to hermitian DPPs: ``FiniteDPP(..., hermitian=True, ...)``.
            - ``"vfx"`` dpp-vfx rejection sampler of :cite:`DeCaVa19`, see :ref:`finite_dpps_exact_sampling_intermediate_sampling_method`. It applies to ``FiniteDPP("likelihood", hermitian=True, L_eval_X_data=(eval_L, X_data))``,
            - ``"alpha"`` alpha-dpp rejection sampler :cite:`CaDeVa20`, see :ref:`finite_dpps_exact_sampling_intermediate_sampling_method`. It applies to ``FiniteDPP("likelihood", hermitian=True, L_eval_X_data=(eval_L, X_data))``.
            - ``"Schur"``, conditionals are computed as Schur complements see :eq:`eq:chain_rule_schur`. It applies to ``FiniteDPP("correlation", projection=True, ...)``.
            - ``"Chol"``, see :ref:`finite_dpps_exact_sampling_cholesky_method`. It refers to :cite:`Pou19` Algorithm 1 (see ``"generic"``) and Algorithm 3 which applies to``FiniteDPP("correlation", projection=True, hermitian=true, ...)``.
            - ``"generic"``, see :ref:`finite_dpps_exact_sampling_cholesky_method`, applies to generic (hermitian or not) finite DPPs, :cite:`Pou19` Algorithm 1.

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
            Returns a sample from the corresponding :class:`FiniteDPP <FiniteDPP>` object. In any case, the sample is appended to the :py:attr:`~FiniteDPP.list_of_samples` attribute as a list.
        :rtype:
            list

        .. note::

            Each time you call this method, the sample is appended to the :py:attr:`~FiniteDPP.list_of_samples` attribute as a list.

            The :py:attr:`~FiniteDPP.list_of_samples` attribute can be emptied using :py:meth:`~FiniteDPP.flush_samples`

        .. seealso::

            - :ref:`finite_dpps_exact_sampling`
            - :py:meth:`~FiniteDPP.flush_samples`
            - :py:meth:`~FiniteDPP.sample_mcmc`
        """

        rng = check_random_state(random_state)
        sampler = self._select_sampler_exact_dpp(method)
        sample = sampler(self, rng, **params)

        self.list_of_samples.append(sample)
        return sample

    def _select_sampler_exact_dpp(self, method):
        samplers = {
            "spectral": spectral_sampler,
            "vfx": vfx_sampler_dpp,
            "alpha": alpha_sampler_dpp,
            "schur": schur_sampler,
            "chol": chol_sampler,
            "generic": generic_sampler,
        }
        default = "spectral" if self.hermitian else "generic"
        return samplers.get(method.lower(), samplers[default])

    def sample_exact_k_dpp(self, size, method="spectral", random_state=None, **params):
        """Sample exactly from :math:`\\operatorname{k-DPP}`. A priori the :class:`FiniteDPP <FiniteDPP>` object was instanciated by its likelihood :math:`\\mathbf{L}` kernel so that

        .. math::

            \\mathbb{P}_{\\operatorname{k-DPP}}(\\mathcal{X} = S)
                \\propto \\det \\mathbf{L}_S ~ 1_{|S|=k}

        :param size:
            size :math:`k` of the :math:`\\operatorname{k-DPP}`

        :type size:
            int

        :param mode:
            - ``projection=True``:
                - ``"GS"`` (default): Gram-Schmidt on the rows of :math:`\\mathbf{K}`.
                - ``"Schur"``: Use Schur complement to compute conditionals.

            - ``projection=False``:
                - ``"GS"`` (default): Gram-Schmidt on the rows of the eigenvectors of :math:`\\mathbf{K}` selected in Phase 1.
                - ``"GS_bis"``: Slight modification of ``"GS"``
                - ``"KuTa12"``: Algorithm 1 in :cite:`KuTa12`
                - ``"vfx"``: the dpp-vfx rejection sampler in :cite:`DeCaVa19`
                - ``"alpha"``: the alpha-dpp rejection sampler in :cite:`CaDeVa20`

        :type mode:
            string, default ``"GS"``

        :param dict params:
            Dictionary containing the parameters for exact samplers with keys

            ``"random_state"`` (default None)

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
            A sample from the corresponding :math:`\\operatorname{k-DPP}`.

            In any case, the sample is appended to the :py:attr:`~FiniteDPP.list_of_samples` attribute as a list.

        :rtype:
            list

        .. note::

            Each time you call this method, the sample is appended to the :py:attr:`~FiniteDPP.list_of_samples` attribute as a list.

            The :py:attr:`~FiniteDPP.list_of_samples` attribute can be emptied using :py:meth:`~FiniteDPP.flush_samples`

        .. caution::

            The underlying kernel :math:`\\mathbf{K}`, resp. :math:`\\mathbf{L}` must be real valued for now.

        .. seealso::

            - :py:meth:`~FiniteDPP.sample_exact`
            - :py:meth:`~FiniteDPP.sample_mcmc_k_dpp`
        """

        rng = check_random_state(random_state)
        sampler = self._select_sampler_exact_k_dpp(method)
        sample = sampler(self, size, rng, **params)

        self.size_k_dpp = size
        self.list_of_samples.append(sample)
        return sample

    @staticmethod
    def _select_sampler_exact_k_dpp(method):
        samplers = {
            "spectral": spectral_sampler_k_dpp,
            "vfx": vfx_sampler_k_dpp,
            "alpha": alpha_sampler_k_dpp,
            "schur": schur_sampler_k_dpp,
            "chol": chol_sampler_k_dpp,
        }
        default = samplers["spectral"]
        return samplers.get(method.lower(), default)

    def sample_mcmc(self, method="aed", random_state=None, **params):
        """Run a MCMC with stationary distribution the corresponding :class:`FiniteDPP <FiniteDPP>` object.

        :param string method:

            - ``"aed"`` add-exchange-delete
            - ``"ad"`` add-delete
            - ``"e"`` exchange
            - ``"zonotope"`` Zonotope sampling

        :param dict params:
            Dictionary containing the parameters for MCMC samplers with keys

            ``"random_state"`` (default None)

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

            In any case, the full trajectory of the Markov chain, made of ``params["nb_iter"]`` samples, is appended to the :py:attr:`~FiniteDPP.list_of_samples` attribute as a list of lists.

        :rtype:
            list

        .. note::

            Each time you call this method, the full trajectory of the Markov chain, made of ``params["nb_iter"]`` samples, is appended to the :py:attr:`~FiniteDPP.list_of_samples` attribute as a list of lists.

            The :py:attr:`~FiniteDPP.list_of_samples` attribute can be emptied using :py:meth:`~FiniteDPP.flush_samples`

        .. seealso::

            - :ref:`finite_dpps_mcmc_sampling`
            - :py:meth:`~FiniteDPP.sample_exact`
            - :py:meth:`~FiniteDPP.flush_samples`
        """

        rng = check_random_state(random_state)
        sampler = self._select_sampler_mcmc_dpp(method)
        chain = sampler(self, rng, **params)

        self.list_of_samples.append(chain)
        return chain[-1]

    @staticmethod
    def _select_sampler_mcmc_dpp(method):
        samplers = {
            "aed": add_exchange_delete_sampler,
            "ad": add_delete_sampler,
            "e": exchange_sampler,
            "zonotope": zonotope_sampler,
        }
        default = samplers["aed"]
        return samplers.get(method.lower(), default)

    def sample_mcmc_k_dpp(self, size, method="e", random_state=None, **params):
        """Calls :py:meth:`~sample_mcmc` with ``mode="E"`` and ``params["size"] = size``

        .. seealso::

            - :ref:`finite_dpps_mcmc_sampling`
            - :py:meth:`~FiniteDPP.sample_mcmc`
            - :py:meth:`~FiniteDPP.sample_exact_k_dpp`
            - :py:meth:`~FiniteDPP.flush_samples`
        """
        self.size_k_dpp = size
        params["size"] = size
        return self.sample_mcmc(method="e", random_state=None, **params)

    def compute_K(self):
        """Alias of :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_correlation_kernel`."""
        return self.compute_correlation_kernel()

    def compute_correlation_kernel(self):
        r"""Compute the correlation kernel :math:`\mathbf{K}` from the current parametrization of the :class:`FiniteDPP` object.

        The returned kernel is also stored as the :py:attr:`~dppy.finite.dpp.FiniteDPP.K` attribute.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """
        while self._compute_correlation_kernel_step():
            continue
        return self.K

    def _compute_correlation_kernel_step(self):
        """Return
        ``False`` if the right parameters are indeed computed
        ``True`` if extra computations are required
        """
        if self.K is not None:
            return False

        if self.K_eig_vals is not None:
            lambda_, U = self.K_eig_vals, self.eig_vecs
            self.K = (U * lambda_).dot(U.T)
            return False

        if self.A_zono is not None:
            rank = self.A_zono.shape[0]
            self.K_eig_vals = np.ones(rank)
            self.eig_vecs, *_ = la.qr(self.A_zono.T, mode="economic")
            return True

        if self.L_eig_vals is not None:
            gamma = self.L_eig_vals
            self.K_eig_vals = gamma / (1.0 + gamma)
            return True

        if self.L is not None:
            # todo separate (non)hermitian cases K = L(L+I)-1
            self.L_eig_vals, self.eig_vecs = la.eigh(self.L)
            return True

        self.compute_likelihood_kernel()
        return True

    def compute_L(self):
        """Alias of :py:meth:`~dppy.finite.dpp.FiniteDPP.compute_likelihood_kernel`"""
        return self.compute_likelihood_kernel()

    def compute_likelihood_kernel(self):
        r"""Compute the likelihood kernel :math:`\mathbf{L}` from the current parametrization of the :class:`FiniteDPP` object.

        The returned kernel is also stored as the :py:attr:`~dppy.finite.dpp.FiniteDPP.L` attribute.

        .. seealso::

            :ref:`finite_dpps_relation_kernels`
        """
        while self._compute_likelihood_kernel_step():
            continue
        return self.L

    def _compute_likelihood_kernel_step(self):
        """Return
        ``False`` if the right parameters are indeed computed
        ``True`` if extra computations are required
        """
        if self.L is not None:
            return False

        if self.projection and self.kernel_type == "correlation":
            raise ValueError(
                "Likelihood kernel L cannot be computed as L = K (I - K)^-1 since projection kernel K has some eigenvalues equal 1"
            )

        if self.L_eig_vals is not None:
            gamma, V = self.L_eig_vals, self.eig_vecs
            self.L = (V * gamma).dot(V.T)
            return False

        if self.L_gram_factor is not None:
            Phi = self.L_gram_factor
            self.L = Phi.T.dot(Phi)
            return False

        if self.eval_L is not None:
            warn_print = [
                "Weird setting:",
                "FiniteDPP(.., **{'L_eval_X_data': (eval_L, X_data)})",
                "When using 'L_eval_X_data', you are a priori working with a big `X_data` and not willing to compute the full likelihood kernel L",
                "Right now, the computation of L=eval_L(X_data) is performed but might be very expensive, this is at your own risk!",
                "You might also use FiniteDPP(.., **{'L': eval_L(X_data)})",
            ]
            warn("\n".join(warn_print))
            self.L = self.eval_L(self.X_data)
            return False

        if self.K_eig_vals is not None:
            try:  # to compute eigenvalues of kernel L = K(I-K)^-1
                np.seterr(divide="raise")
                self.L_eig_vals = self.K_eig_vals / (1.0 - self.K_eig_vals)
                return True
            except FloatingPointError:
                err_print = [
                    "Eigenvalues of the likelihood L kernel cannot be computed as eig_L = eig_K / (1 - eig_K).",
                    "K kernel has some eig_K very close to 1. Hint: `K` kernel might be a projection",
                ]
                raise FloatingPointError("\n".join(err_print))

        if self.K is not None:
            # todo separate (non)hermitian cases L = K(K-I)-1
            eig_vals, self.eig_vecs = la.eigh(self.K)
            np.clip(eig_vals, 0.0, 1.0, out=eig_vals)  # 0 <= K <= I
            self.K_eig_vals = eig_vals
            return True

        self.compute_correlation_kernel()
        return True

    def plot_kernel(self, kernel_type="correlation", save_path=""):
        """Display a heatmap of the kernel used to define the :class:`FiniteDPP` object (correlation kernel :math:`\\mathbf{K}` or likelihood kernel :math:`\\mathbf{L}`)

        :param str kernel_type: Type of kernel (``"correlation"`` or ``"likelihood"``), default ``"correlation"``

        :param str save_path: Path to save plot, if empty (default) the plot is not saved.
        """

        if not kernel_type:
            kernel_type = self.kernel_type

        fig, ax = plt.subplots(1, 1)

        if kernel_type == "correlation":
            self.compute_K()
            nb_items, kernel_to_plot = self.K.shape[0], self.K

        elif kernel_type == "likelihood":
            self.compute_L()
            nb_items, kernel_to_plot = self.L.shape[0], self.L

        else:
            raise ValueError("kernel_type != 'correlation' or 'likelihood'")

        heatmap = ax.pcolor(kernel_to_plot, cmap="jet", vmin=-0.3, vmax=1)

        ax.set_aspect("equal")

        ticks = np.arange(nb_items)
        ticks_label = [r"${}$".format(tic) for tic in ticks]

        ax.xaxis.tick_top()
        ax.set_xticks(ticks + 0.5, minor=False)

        ax.invert_yaxis()
        ax.set_yticks(ticks + 0.5, minor=False)

        ax.set_xticklabels(ticks_label, minor=False)
        ax.set_yticklabels(ticks_label, minor=False)

        plt.colorbar(heatmap)

        if save_path:
            plt.savefig(save_path)
