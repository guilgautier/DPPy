from dppy.finite.exact_samplers.intermediate_sampler import (
    intermediate_sampler_dpp,
    intermediate_sampler_k_dpp,
)
from dppy.finite.exact_samplers.projection_sampler_eigen import projection_sampler_eigen
from dppy.finite.exact_samplers.projection_sampler_kernel import (
    projection_sampler_kernel,
)
from dppy.finite.exact_samplers.sequential_sampler import sequential_sampler
from dppy.finite.exact_samplers.spectral_sampler import (
    spectral_sampler_dpp,
    spectral_sampler_k_dpp,
)


def select_exact_sampler_dpp(dpp, method):
    r"""Select exact :math:`\operatorname{DPP}` sampler for ``dpp`` by its name ``method``.

    :param dpp:
        Finite DPP
    :type dpp:
        :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param method: name of the sampling method

        - ``"spectral"``: :py:func:`~dppy.finite.exact_samplers.spectral_sampler.spectral_sampler_dpp`,
        - ``"projection"``: :py:func:`~dppy.finite.exact_samplers.projection_sampler.projection_sampler_eigen` if ``dpp.eig_vecs`` is not None, otherwise :py:func:`~dppy.finite.exact_samplers.projection_sampler.projection_sampler_kernel`,
        - ``"intermediate"``: :py:func:`~dppy.finite.exact_samplers.intermediate_sampler.intermediate_sampler_dpp`,
        - ``"sequential"``: :py:func:`~dppy.finite.exact_samplers.sequential_sampler.sequential_sampler`.

    :type method: string
    :return: sampler
    :rtype: callable
    """
    samplers = {
        "spectral": spectral_sampler_dpp,
        "projection": projection_sampler_eigen
        if dpp.eig_vecs is not None
        else projection_sampler_kernel,
        "intermediate": intermediate_sampler_dpp,
        "sequential": sequential_sampler,
    }
    default = samplers["spectral" if dpp.hermitian else "sequential"]
    return samplers.get(method.lower(), default)


def select_exact_sampler_k_dpp(dpp, method):
    r"""Select exact :math:`k\!\operatorname{-DPP}` sampler for ``dpp`` by its name ``method``.

    :param dpp:
        Finite DPP
    :type dpp:
        :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param method: name of the sampling method

        - ``"spectral"``: :py:func:`~dppy.finite.exact_samplers.spectral_sampler.spectral_sampler_k_dpp`,
        - ``"projection"``: :py:func:`~dppy.finite.exact_samplers.projection_sampler.projection_sampler_eigen` if ``dpp.eig_vecs`` is not None, otherwise :py:func:`~dppy.finite.exact_samplers.projection_sampler.projection_sampler_kernel`,
        - ``"intermediate"``: :py:func:`~dppy.finite.exact_samplers.intermediate_sampler.intermediate_sampler_k_dpp`,

    :type method: string
    :return: sampler
    :rtype: callable
    """
    samplers = {
        "spectral": spectral_sampler_k_dpp,
        "projection": projection_sampler_eigen
        if dpp.eig_vecs is not None
        else projection_sampler_kernel,
        "intermediate": intermediate_sampler_k_dpp,
    }
    default = samplers["spectral"]
    return samplers.get(method.lower(), default)
