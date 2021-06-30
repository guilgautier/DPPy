import numpy as np
from .intermediate_sampling import (
    vfx_sampling_precompute_constants,
    vfx_sampling_do_sampling_loop,
)
from ..utils import check_random_state


def vfx_sampler(dpp, rng, **params):
    if dpp.eval_L is None or dpp.X_data is None:
        raise ValueError(
            "The vfx sampler is currently only available with "
            '{"L_eval_X_data": (L_eval, X_data)} representation.'
        )

    r_state_outer = None
    if "random_state" in params:
        r_state_outer = params.pop("random_state", None)

    sample, dpp.intermediate_sample_info = dpp_vfx_sampler(
        dpp.intermediate_sample_info, dpp.X_data, dpp.eval_L, random_state=rng, **params
    )
    if r_state_outer:
        params["random_state"] = r_state_outer

    return sample


def dpp_vfx_sampler(info, X_data, eval_L, random_state=None, **params):
    """First pre-compute quantities necessary for the vfx rejection sampling loop, such as the inner Nyström approximation, and the RLS of all elements in :math:`\\mathbf{L}`.
    Then, given the pre-computed information,run a rejection sampling loop to generate DPP samples.

    :param info:
        If available, the pre-computed information necessary for the vfx rejection sampling loop.
        If ``None``, this function will compute and return an ``_IntermediateSampleInfo`` with fields

        - ``.alpha_star``: appropriate rescaling such that the expected sample size of :math:`\\operatorname{DPP}(\\alpha^* \\mathbf{L})` is equal to a user-indicated constant ``params['desired_expected_size']``, or 1.0 if no such constant was specified by the user.
        - ``.logdet_I_A``: :math:`\\log \\det` of the Nyström approximation of :math:`\\mathbf{L} + I`
        - ``.q``: placeholder q constant used for vfx sampling, to be replaced by the user before the sampling loop
        - ``.s`` and ``.z``: approximations of the expected sample size of :math:`\\operatorname{DPP}(\\alpha^* \\mathbf{L})` to be used in the sampling loop. For more details see :cite:`DeCaVa19`
        - ``.rls_estimate``: approximations of the RLS of all elements in X (i.e. in :math:`\\mathbf{L}`)

    :type info:
        ``_IntermediateSampleInfo`` or ``None``, default ``None``

    :param array_like X_data:
        dataset such that :math:`\\mathbf{L}=` ``eval_L(X_data)``, out of which we are sampling objects according to a DPP

    :param callable eval_L:
        Likelihood function.
        Given two sets of n points X and m points Y, ``eval_L(X, Y)`` should compute the :math:`n x m` matrix containing the likelihood between points.
        The function should also accept a single argument X and return ``eval_L(X) = eval_L(X, X)``.
        As an example, see the implementation of any of the kernels provided by scikit-learn (e.g. `PairwiseKernel <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.PairwiseKernel.html>`_).

    :param random_state:
        random source used for sampling, if None a RandomState is automatically generated

    :type random_state:
        RandomState or None, default None

    :param dict params:
        Dictionary including optional parameters:

        - ``'desired_expected_size'`` (float or None, default None)

            Desired expected sample size for the DPP.
            If None, use the natural DPP expected sample size.
            The vfx sampling algorithm can approximately adjust the expected sample size of the DPP by rescaling the :math:`\\mathbf{L}` matrix with a scalar :math:`\\alpha^*\\leq 1` .
            Adjusting the expected sample size can be useful to control downstream complexity, and it is necessary to improve the probability of drawing a sample with exactly :math:`k` elements when using vfx for k-DPP sampling.
            Currently only reducing the sample size is supported, and the sampler will return an exception if the DPP sample has already a natural expected size smaller than ``params['desired_expected_size'``.]

        - ``'rls_oversample_dppvfx'`` (float, default 4.0)

            Oversampling parameter used to construct dppvfx's internal Nyström approximation.
            The ``rls_oversample_dppvfx``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_dppvfx`` factor.
            This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_dppvfx``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate.

        - ``'rls_oversample_bless'`` (float, default 4.0)
            Oversampling parameter used during bless's internal Nyström approximation.
            Note that this is a different Nyström approximation than the one related to :func:`rls_oversample_dppvfx`, and can be tuned separately.
            The ``rls_oversample_bless``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_bless`` factor.
            This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_bless``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate or is not accurate.

        - ``'q_func'`` (function, default x: x*x)
            Mapping from estimate expected size of the DPP to Poisson intensity used to choose size of the intermediate sample.
            Larger intermediate sampler cause less efficient iterations but higher acceptance probability.

        - ``'nb_iter_bless'`` (int or None, default None)

            Iterations for inner BLESS execution, if None it is set to log(n)

        - ``'verbose'`` (bool, default True)

            Controls verbosity of debug output, including progress bars.
            If info is not provided, the first progress bar reports the inner execution of
            the bless algorithm, showing:

                - lam: lambda value of the current iteration
                - m: current size of the dictionary (number of centers contained)
                - m_expected: expected size of the dictionary before sampling
                - probs_dist: (mean, max, min) of the approximate rlss at the current iteration

            Subsequent progress bars show the execution of each rejection sampling loops (i.e. once per sample generated)
                - acc_thresh: latest computed probability of acceptance
                - rej_iter: iteration of the rejection sampling loop (i.e. rejections so far)

        - ``'max_iter'`` (int, default 1000)

            Maximum number of intermediate sample rejections before giving up.

    :return:
        Sample from a DPP (as a list) and updated info

    :rtype:
        tuple(list, _IntermediateSampleInfo)
    """
    rng = check_random_state(random_state)

    if info is None:
        info = vfx_sampling_precompute_constants(
            X_data=X_data, eval_L=eval_L, rng=rng, **params
        )

        q_func = params.get("q_func", lambda s: s * s)
        info = info._replace(q=q_func(info.s))

    sample, rej_count = vfx_sampling_do_sampling_loop(
        X_data, eval_L, info, rng, **params
    )

    return sample, info


def k_dpp_vfx_sampler(size, info, X_data, eval_L, random_state=None, **params):
    """First pre-compute quantities necessary for the vfx rejection sampling loop, such as the inner Nyström approximation, and the RLS of all elements in :math:`\\mathbf{L}`.
    Then, given the pre-computed information,run a rejection sampling loop to generate DPP samples.
    To guarantee that the returned sample has size ``size``, we internally set desired_expected_size=size and
    then repeatedly invoke dpp_vfx_sampler until a sample of the correct size is returned,
    or exit with an error after a chosen number of rejections is reached.

    :param int size: The size of the sample (i.e. the k of k-DPPs)

    :param info:
        If available, the pre-computed information necessary for the vfx rejection sampling loop.
        If ``None``, this function will compute and return an ``_IntermediateSampleInfo`` with fields

        - ``.alpha_star``: appropriate rescaling such that the expected sample size of :math:`\\operatorname{DPP}(\\alpha^* \\mathbf{L})` is equal to a user-indicated constant ``params['desired_expected_size']``, or 1.0 if no such constant was specified by the user.
        - ``.logdet_I_A``: :math:`\\log \\det` of the Nyström approximation of :math:`\\mathbf{L} + I`
        - ``.q``: placeholder q constant used for vfx sampling, to be replaced by the user before the sampling loop
        - ``.s`` and ``.z``: approximations of the expected sample size of :math:`\\operatorname{DPP}(\\alpha^* \\mathbf{L})` to be used in the sampling loop. For more details see :cite:`DeCaVa19`
        - ``.rls_estimate``: approximations of the RLS of all elements in X (i.e. in :math:`\\mathbf{L}`)

    :type info:
        ``_IntermediateSampleInfo`` or ``None``, default ``None``

    :param array_like X_data:
        dataset such that :math:`\\mathbf{L}=` ``eval_L(X_data)``, out of which we are sampling objects according to a DPP

    :param callable eval_L:
        Likelihood function.
        Given two sets of n points X and m points Y, ``eval_L(X, Y)`` should compute the :math:`n x m` matrix containing the likelihood between points.
        The function should also accept a single argument X and return ``eval_L(X) = eval_L(X, X)``.
        As an example, see the implementation of any of the kernels provided by scikit-learn (e.g. `PairwiseKernel <https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.PairwiseKernel.html>`_).

    :param random_state:
        random source used for sampling, if None a RandomState is automatically generated

    :type random_state:
        RandomState or None, default None

    :param dict params:
        Dictionary including optional parameters:

        - ``'rls_oversample_dppvfx'`` (float, default 4.0)

            Oversampling parameter used to construct dppvfx's internal Nyström approximation.
            The ``rls_oversample_dppvfx``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_dppvfx`` factor.
            This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_dppvfx``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate.

        - ``'rls_oversample_bless'`` (float, default 4.0)
            Oversampling parameter used during bless's internal Nyström approximation.
            Note that this is a different Nyström approximation than the one related to :func:`rls_oversample_dppvfx`, and can be tuned separately.
            The ``rls_oversample_bless``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_bless`` factor.
            This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_bless``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate or is not accurate.

        - ``'q_func'`` (function, default x: x*x)
            Mapping from estimate expected size of the DPP to Poisson intensity used to choose size of the intermediate sample.
            Larger intermediate sampler cause less efficient iterations but higher acceptance probability.

        - ``'nb_iter_bless'`` (int or None, default None)

            Iterations for inner BLESS execution, if None it is set to log(n)

        - ``'verbose'`` (bool, default True)

            Controls verbosity of debug output, including progress bars.
            If info is not provided, the first progress bar reports the inner execution of
            the bless algorithm, showing:

                - lam: lambda value of the current iteration
                - m: current size of the dictionary (number of centers contained)
                - m_expected: expected size of the dictionary before sampling
                - probs_dist: (mean, max, min) of the approximate rlss at the current iteration

            Subsequent progress bars show the execution of each rejection sampling loops (i.e. once per sample generated)
                - acc_thresh: latest computed probability of acceptance
                - rej_iter: iteration of the rejection sampling loop (i.e. rejections so far)

        - ``'max_iter'`` (int, default 1000)

            Maximum number of intermediate sample rejections before giving up.

        - ``'max_iter_size_rejection'`` (int, default 100)

            Maximum number of size-based rejections before giving up.


    :return:
        Sample from a DPP (as a list) and updated info

    :rtype:
        tuple(list, _IntermediateSampleInfo)
    """
    rng = check_random_state(random_state)

    if (info is None) or (not np.isclose(info.s, size).item()):
        info = vfx_sampling_precompute_constants(
            X_data=X_data, eval_L=eval_L, desired_expected_size=size, rng=rng, **params
        )

        q_func = params.get("q_func", lambda s: s * s)
        info = info._replace(q=q_func(info.s))

    max_iter_size_rejection = params.get("max_iter_size_rejection", 100)

    for _ in range(max_iter_size_rejection):
        sample, rej_count = vfx_sampling_do_sampling_loop(
            X_data, eval_L, info, rng, **params
        )

        tmp = info.rej_to_first_sample + rej_count
        info = info._replace(rej_to_first_sample=tmp)
        if len(sample) == size:
            break
    else:
        raise ValueError(
            "The vfx sampler reached the maximum number of rejections allowed "
            "for the k-DPP size rejection ({}), try to increase the q factor "
            "(see q_func parameter) or the Nyström approximation accuracy "
            "see rls_oversample_* parameters).".format(max_iter_size_rejection)
        )

    return sample, info
