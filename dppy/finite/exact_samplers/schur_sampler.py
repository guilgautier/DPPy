from dppy.finite.exact_samplers.projection_kernel_samplers import (
    projection_kernel_sampler,
)


def schur_sampler(dpp, random_state=None, **params):
    r"""Generate an exact sample from a projection ``dpp`` using the :ref:`chain rule <finite_dpps_exact_sampling_projection_dpp_chain_rule>` for :math:`k` steps.

    :param dpp: Finite projection DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`
    :param random_state: random number generator or seed, defaults to None
    :type random_state: optional

    Keyword arguments:
        - mode (str): select the variant of the sampler, see :py:func:`~dppy.finite.exact_samplers.projection_kernel_samplers.projection_kernel_sampler`

    :return: sample
    :rtype: list
    """
    params["mode"] = "schur"
    return projection_kernel_sampler(dpp, random_state=random_state, **params)


def schur_sampler_k_dpp(dpp, size, random_state=None, **params):
    r"""Generate an exact sample from a projection :math:`k\!\operatorname{-DPP}` associated with ``dpp`` and :math:`k=` ``size``, using the :ref:`chain rule <finite_dpps_exact_sampling_projection_dpp_chain_rule>` for :math:`k` steps.

    Calls :py:func:`~dppy.finite.exact_samplers.schur_sampler` with extra keyword argument ``size=size``.

    :param dpp: Finite projection DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param size: size :math:`k` of the output sample
    :type size: int

    :param random_state: random number generator or seed, defaults to None
    :type random_state: optional

    Keyword arguments:
        - mode (str): select the variant of the sampler, see :py:func:`~dppy.finite.exact_samplers.projection_kernel_samplers.projection_kernel_sampler`

    :return: sample
    :rtype: list
    """
    params["size"] = size
    return schur_sampler(dpp, random_state=None, **params)
