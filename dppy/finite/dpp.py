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
    r"""Object representing :ref:`finite determinantal point process <finite_dpps_definition>`, with attributes.

    :ivar kernel_type: initial value: ``kernel_type``.
    :ivar projection: initial value: ``projection``.
    :ivar hermitian: initial value: ``hermitian``.

    :ivar K: initial value: keyword argument ``K`` (default None).
    :ivar K_eig_vals: initial value: keyword argument ``K_eig_dec[0]`` (default None).
    :ivar A_zono: initial value: keyword argument ``A_zono`` (default None).

    :ivar L: initial value: keyword argument ``L`` (default None).
    :ivar L_eig_vals: initial value: keyword argument ``L_eig_dec[0]`` (default None).
    :ivar L_gram_factor: initial value: keyword argument ``L_gram_factor`` (default None).

    :ivar eig_vecs: initial value: keyword argument ``K_eig_dec[1]`` or ``L_eig_dec[1]`` (default None).

    :ivar eval_L: initial value: keyword argument ``L_eval_X_data[0]`` (default None).
    :ivar X_data: initial value: keyword argument ``L_eval_X_data[1]`` (default None).

    :ivar list_of_samples: initial value: ``[]``.
    :ivar size_k_dpp: initial value: ``0``.
    :ivar esp: initial value: ``None``.
    :ivar intermediate_sample_info: initial value: ``None``.
    """

    def __init__(self, kernel_type, projection=False, hermitian=True, **params):
        r"""Instantiate a :py:class:`~dppy.finite.FiniteDPP` object form

        :param string kernel_type:
            Indicate if the associated :math:`\operatorname{DPP}` is defined via its

            - ``"correlation"`` :math:`\mathbf{K}` kernel, or
            - ``"likelihood"`` :math:`\mathbf{L}` kernel.

        :param projection:
            Indicate if the associated kernel is of projection type, i.e., :math:`M^2 = M`.
        :type projection:
            bool, default ``False``.

        :param hermitian:
            Indicate if the associated kernel is hermitian, i.e., :math:`M^* = M`.
        :type hermitian:
            bool, default ``True``.

        :Keyword Arguments:

            If ``kernel_type="correlation"``

            - **K** (numpy.ndarray) -- correlation kernel :math:`\mathbf{K}` of size :math:`N \times N`. If ``hermitian=True`` then :math:`0 \preceq \mathbf{K} \preceq I` must be satisfied.

            - **K_eig_dec** (tuple(numpy.ndarray, numpy.ndarray)) -- ``(eig_vals, eig_vecs)`` Eigendecomposition of the correlation kernel :math:`\mathbf{K}=U\Lambda U^*`, such that :math:`U=` ``eig_vecs``, :math:`\Lambda=\operatorname{diag}` (``eig_vals``) with :math:`0 \leq` ``eig_vals`` :math:`\leq 1`. Applies only if ``hermitian=True``.

            - **A_zono** (numpy.ndarray) -- Matrix of size :math:`d \times N` such that :math:`\operatorname{rank}(A)=d` and :math:`\mathbf{K} = A^{\top} (A A^{\top})^{-1} A`.

            If ``kernel_type="likelihood"``

            - **L** (numpy.ndarray) -- likelihood kernel :math:`\mathbf{L}` of size :math:`N \times N`. If ``hermitian=True`` then :math:`\mathbf{L} \succeq 0` must be satisfied.

            - **L_eig_dec** (tuple(numpy.ndarray, numpy.ndarray)) -- ``(eig_vals, eig_vecs)``.  Eigendecomposition of the likelihood kernel :math:`\mathbf{L}=V\Gamma V^*`, such that :math:`V=` ``eig_vecs``, :math:`\Gamma=\operatorname{diag}` (``eig_vals``), with ``eig_vals`` :math:`\geq 0`. Applies only if ``hermitian=True``.

            - **L_gram_factor** (numpy.ndarray) -- Matrix :math:`\Phi` of size :math:`d \times N` such that :math:`\mathbf{L} = \Phi^{\top} \Phi`.

            - **L_eval_X_data** (tuple(callable, numpy.ndarray)) -- ``(L, X)`` such that :math:`\mathbf{L}_{ij} =` ``L(X[i, :], X[j, :])``. For a full description of the requirements imposed on ``L``'s interface, see the documentation of :func:`~dppy.finite.exact_samplers.vfx_samplers.vfx_sampling_precompute_constants`. For an example, see the implementation of any of the kernels provided by scikit-learn (e.g. sklearn.gaussian_process.kernels.PairwiseKernel).

        :type params:
            dict
        """

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

    def sample_exact(self, method="", random_state=None, **kwargs):
        """Sample exactly from the corresponding :py:class:`~dppy.finite.dpp.FiniteDPP` object.

        :param method:

            - ``"sequential"``. It corresponds to a generic sampler, which applies to any valid DPP (hermitian or not), see :ref:`finite_dpps_exact_sampling_sequential_methods` and :py:func:`~dppy.finite.exact_samplers.sequential_samplers.sequential_sampler`.

            - ``"spectral"``. It applies only if the attribute :py:attr:`~dppy.finite.dpp.FiniteDPP.hermitian` is True, see :ref:`finite_dpps_exact_sampling_spectral_method` and :py:func:`~dppy.finite.exact_samplers.spectral_samplers.spectral_sampler_dpp`.

            - ``"intermediate"``. It applies only if the attribute :py:attr:`~dppy.finite.dpp.FiniteDPP.hermitian` is True, see :ref:`finite_dpps_exact_sampling_intermediate_sampling_methods` and :py:func:`~dppy.finite.exact_samplers.intermediate_samplers.intermediate_sampler_dpp`.

            - ``"projection"``. It applies only if the attribute :py:attr:`~dppy.finite.dpp.FiniteDPP.projection` is True, see :ref:`finite_dpps_exact_sampling_projection_dpp` and :py:func:`~dppy.finite.exact_samplers.projection_samplers.projection_sampler_dpp`.

        :type method:
            string, default ``"spectral"`` if the attribute :py:attr:`~dppy.finite.dpp.FiniteDPP.hermitian` is True, otherwise ``"sequential"``.

        :Keyword Arguments:
            Please refer to the documentation of the sampler associated to the ``method`` argument.

        :return:
            Returns a sample from the corresponding :py:class:`~dppy.finite.dpp.FiniteDPP` object. In any case, the sample is appended to the :py:attr:`~dpp.finite.dpp.FiniteDPP.list_of_samples` attribute as a list.
        :rtype:
            list

        .. seealso::

            - :ref:`finite_dpps_exact_sampling`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.flush_samples`
            - :py:meth:`~dppy.finite.dpp.FiniteDPP.sample_mcmc`
        """
        rng = check_random_state(random_state)
        sampler = select_sampler_exact_dpp(self, method)
        sample = sampler(self, rng, **kwargs)

        self.list_of_samples.append(sample)
        return sample

    def sample_exact_k_dpp(self, size, method="spectral", random_state=None, **kwargs):
        r"""Sample exactly from the corresponding :math:`\operatorname{k-DPP}` with :math:`k=` ``size``, see :ref:`finite_dpps_exact_sampling_k_dpps`.

        It applies only if the attribute :py:attr:`~dppy.finite.dpp.FiniteDPP.hermitian` is True.

        :param method:

            - ``"spectral"`` (default), see :ref:`finite_dpps_exact_sampling_k_dpps` and :py:func:`~dppy.finite.exact_samplers.spectral_samplers.spectral_sampler_k_dpp`.

            - ``"intermediate"``, see :ref:`finite_dpps_exact_sampling_intermediate_sampling_methods` and :py:func:`~dppy.finite.exact_samplers.intermediate_samplers.intermediate_sampler_k_dpp`.

            - ``"projection"``. It applies only if the attribute :py:attr:`~dppy.finite.dpp.FiniteDPP.projection` is True, see  :py:func:`~dppy.finite.exact_samplers.projection_samplers.projection_sampler_k_dpp`.

        :type method:
            string, default ``"spectral"``.

        :Keyword Arguments:
            Please refer to the documentation of the sampler associated to the ``method`` argument.

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
        sample = sampler(self, size, rng, **kwargs)

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
