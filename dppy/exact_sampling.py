# coding: utf8
"""Implementation of finite DPP exact samplers derived from:

- the raw **projection** correlation :math:`K` kernel (no need for eigendecomposition)
- the eigendecomposition of the correlation :math:`K` kernel

.. seealso:

    `Documentation on ReadTheDocs <https://dppy.readthedocs.io/en/latest/finite_dpps/exact_sampling.html>`_
"""

import numpy as np
import scipy.linalg as la
from dppy.utils import inner1d, check_random_state, get_progress_bar
from dppy.intermediate_sampling import (
    vfx_sampling_precompute_constants,
    vfx_sampling_do_sampling_loop,
    alpha_dpp_sampling_precompute_constants,
    alpha_dpp_sampling_do_sampling_loop,
)


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


def alpha_dpp_sampler(info, X_data, eval_L, random_state=None, **params):
    """First pre-compute quantities necessary for the alpha-dpp rejection sampling loop, such as the inner Nyström
    approximation, and the and the initial rescaling alpha_hat for the binary search.
    Then, given the pre-computed information,run a rejection sampling loop to generate samples from DPP(alpha * L).

    :param info:
        If available, the pre-computed information necessary for the alpha-dpp rejection sampling loop.
        If ``None``, this function will compute and return an ``_IntermediateSampleInfoAlphaRescale`` (see :func:`alpha_dpp_sampling_precompute_constants`)

    :type info:
        ``_IntermediateSampleInfoAlphaRescale`` or ``None``, default ``None``

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

            Desired expected sample size for the rescaled DPP.
            If None, use the natural DPP expected sample size.
            The alpha-dpp sampling algorithm can approximately adjust the expected sample size of the DPP by rescaling the :math:`\\mathbf{L}` matrix with a scalar :math:`\\alpha^*\\leq 1` .
            Adjusting the expected sample size can be useful to control downstream complexity, and it is necessary to improve the probability of drawing a sample with exactly :math:`k` elements when using alpha-dpp for k-DPP sampling.
            Currently only reducing the sample size is supported, and the sampler will return an exception if the DPP sample has already a natural expected size smaller than ``params['desired_expected_size'``.]

        - ``'rls_oversample_alphadpp'`` (float, default 4.0)

            Oversampling parameter used to construct alphadpp's internal Nyström approximation.
            The ``rls_oversample_alphadpp``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_alphadpp`` factor.
            This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_alphadpp``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate.

        - ``'rls_oversample_bless'`` (float, default 4.0)
            Oversampling parameter used during bless's internal Nyström approximation.
            Note that this is a different Nyström approximation than the one related to :func:`rls_oversample_alphadpp`, and can be tuned separately.
            The ``rls_oversample_bless``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_bless`` factor.
            This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_bless``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate or is not accurate.

        - ``'r_func'`` (function, default x: x)
            Mapping from estimate expected size of the rescaled alpha-DPP to Poisson intensity used to choose size
            of the intermediate sample. Larger intermediate sampler cause less efficient iterations but higher
            acceptance probability.

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
        tuple(list, _IntermediateSampleInfoAlphaRescale)
    """
    rng = check_random_state(random_state)

    if info is None:
        info = alpha_dpp_sampling_precompute_constants(
            X_data=X_data, eval_L=eval_L, rng=rng, **params
        )

        r_func = params.get("r_func", lambda r: r)
        info = info._replace(r=r_func(info.deff_alpha_L_hat))

    sample, rej_count, info = alpha_dpp_sampling_do_sampling_loop(
        X_data, eval_L, info, rng, **params
    )

    return sample, info


##########
# k-DPPs #
##########


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


def alpha_k_dpp_sampler(size, info, X_data, eval_L, random_state=None, **params):
    """First pre-compute quantities necessary for the alpha-dpp rejection sampling loop, such as the inner Nyström
    approximation, the and the initial rescaling alpha_hat for the binary search.
    Then, given the pre-computed information,run a rejection sampling loop to generate k-DPP samples.
    To guarantee that the returned sample has size ``size``, we internally set desired_expected_size=size and
    then repeatedly invoke alpha_dpp_sampler until a sample of the correct size is returned,
    or exit with an error after a chosen number of rejections is reached.

    :param int size: The size of the sample (i.e. the k of k-DPPs)

    :param info:
        If available, the pre-computed information necessary for the alpha-dpp rejection sampling loop.
        If ``None``, this function will compute and return an ``_IntermediateSampleInfoAlphaRescale`` (see :func:`alpha_dpp_sampling_precompute_constants`)

    :type info:
        ``_IntermediateSampleInfoAlphaRescale`` or ``None``, default ``None``

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

        - ``'rls_oversample_alphadpp'`` (float, default 4.0)

            Oversampling parameter used to construct alphadpp's internal Nyström approximation.
            The ``rls_oversample_alphadpp``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_alphadpp`` factor.
            This makes each rejection round slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_alphadpp``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate.

        - ``'rls_oversample_bless'`` (float, default 4.0)
            Oversampling parameter used during bless's internal Nyström approximation.
            Note that this is a different Nyström approximation than the one related to :func:`rls_oversample_alphadpp`, and can be tuned separately.
            The ``rls_oversample_bless``:math:`\\geq 1` parameter is used to increase the rank of the approximation by a ``rls_oversample_bless`` factor.
            This makes the one-time pre-processing slower and more memory intensive, but reduces variance and the number of rounds of rejections, so the actual runtime might increase or decrease.
            Empirically, a small factor ``rls_oversample_bless``:math:`\\in [2,10]` seems to work.
            It is suggested to start with a small number and increase if the algorithm fails to terminate or is not accurate.

        - ``'r_func'`` (function, default x: x)
            Mapping from estimate expected size of the rescaled alpha-DPP to Poisson intensity used to choose size of the intermediate sample.
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

        - ``'early_stop'`` (bool, default False)

            Wheter to return as soon as a first sample is accepted. If True, the sampling loop is interrupted as soon as a k-DPP sample is generated.
            If False, the algorithm continues the binary search until of a sufficiently good rescaling alpha is found.
            While this makes subsequent sampling faster, it is wasteful in the case where a single k-DPP sample is desired.

        - ``'max_iter_size_rejection'`` (int, default 100)

            Maximum number of size-based rejections before giving up.

        - ``'max_iter_size_rejection'`` (int, default 100)

            Maximum number of size-based rejections before giving up.


    :return:
        Sample from a DPP (as a list) and updated info

    :rtype:
        tuple(list, _IntermediateSampleInfoAlphaRescale)
    """
    rng = check_random_state(random_state)

    if info is None or info.k != size:
        info = alpha_dpp_sampling_precompute_constants(
            X_data=X_data, eval_L=eval_L, desired_expected_size=size, rng=rng, **params
        )

        r_func = params.get("r_func", lambda r: r)

        info = info._replace(r=r_func(info.deff_alpha_L_hat))

    max_iter_size_rejection = params.get("max_iter_size_rejection", 100)
    number_trial_search = np.ceil(np.sqrt(size)).astype("int")
    stopping_ratio = 1 + 1 / (size + 3) ** 2

    sample_count = 0
    trial_count = 0
    under_k_count = 0
    over_k_count = 0

    ratio_alpha = info.alpha_max / info.alpha_min
    found_good_alpha = ratio_alpha <= stopping_ratio

    prog_bar = get_progress_bar(disable=not params.get("verbose", False))
    verbose_outer = None
    if "verbose" in params:
        verbose_outer = params.pop("verbose")
    params["verbose"] = False

    early_stop = params.get("early_stop", False)

    trial_count_overall = 0
    for _ in range(max_iter_size_rejection):
        sample, rej_count, info = alpha_dpp_sampling_do_sampling_loop(
            X_data, eval_L, info, rng, **params
        )

        trial_count += 1
        trial_count_overall += 1

        prog_bar.set_postfix(
            trial_count=trial_count,
            alpha="{:.4}".format(info.alpha_hat),
            alpha_switch=info.alpha_switches,
            k=size,
            k_emp=len(sample),
            rej_count=rej_count,
        )
        prog_bar.update()

        if len(sample) == size:
            sample_out = sample
            if info.trial_to_first_sample == 0:
                info = info._replace(trial_to_first_sample=trial_count_overall)
            sample_count += 1
            if early_stop:
                break

        under_k_count += len(sample) < size
        over_k_count += len(sample) > size

        if info.trial_to_first_sample == 0:
            tmp = info.rej_to_first_sample + rej_count
            info = info._replace(rej_to_first_sample=tmp)

        if sample_count == 2:
            found_good_alpha = True
            break

        if trial_count == number_trial_search:
            if under_k_count > over_k_count:
                info = info._replace(alpha_min=info.alpha_hat)
            else:
                info = info._replace(alpha_max=info.alpha_hat)

            geom_mean_alpha = np.sqrt(info.alpha_min * info.alpha_max)
            info = info._replace(alpha_hat=geom_mean_alpha)
            diag_L = info.diag_L
            info = info._replace(rls_upper_bound=geom_mean_alpha * diag_L)
            rls_ub_valid = np.full((diag_L.shape[0],), False)
            info = info._replace(rls_upper_bound_valid=rls_ub_valid)

            ratio_alpha = info.alpha_max / info.alpha_min
            found_good_alpha = ratio_alpha <= stopping_ratio and sample_count > 0
            if found_good_alpha:
                break

            info = info._replace(alpha_switches=info.alpha_switches + 1)
            trial_count = 0
            under_k_count = 0
            over_k_count = 0
    else:
        raise ValueError(
            "The alpha sampler reached the maximum number of rejections allowed "
            "for the k-DPP size rejection ({}), try to increase the r factor "
            "(see r_func parameter) or the Nyström approximation accuracy "
            "see rls_oversample_* parameters).".format(max_iter_size_rejection)
        )
    if found_good_alpha:
        info = info._replace(alpha_min=info.alpha_hat)
        info = info._replace(alpha_max=info.alpha_hat)
        info = info._replace(alpha_switches=info.alpha_switches + 1)

    if verbose_outer:
        params["verbose"] = verbose_outer
    else:
        params.pop("verbose")

    return sample_out, info


def k_dpp_eig_vecs_selector(eig_vals, eig_vecs, size, esp=None, random_state=None):
    """Select columns of ``eig_vecs`` by sampling Bernoulli variables with parameters derived from the computation of elementary symmetric polynomials ``esp`` of order 0 to ``size`` evaluated in ``eig_vals``.
    This corresponds to :cite:`KuTa12` Algorithm 8.

    :param eig_vals:
        Collection of eigenvalues (assumed non-negetive)
    :type eig_vals:
        array_like

    :param eig_vecs:
        Matrix of eigenvectors stored columnwise
    :type eig_vecs:
        array_like

    :param size:
        Number of eigenvectors to be selected
    :type size:
        int

    :param esp:
        Computation of the elementary symmetric polynomials previously evaluated in ``eig_vals`` and returned by :py:func:`elementary_symmetric_polynomials <elementary_symmetric_polynomials>`, default to None.
    :type esp:
        array_like

    :return:
        Selected eigenvectors
    :rtype:
        array_like

    .. seealso::

        - :cite:`KuTa12` Algorithm 8
        - :func:`elementary_symmetric_polynomials <elementary_symmetric_polynomials>`
    """

    rng = check_random_state(random_state)

    # Size of: ground set / sample
    N, k = eig_vecs.shape[0], size

    # as in np.linalg.matrix_rank
    tol = np.max(eig_vals) * N * np.finfo(float).eps
    rank = np.count_nonzero(eig_vals > tol)
    if k > rank:
        raise ValueError("size k={} > rank={}".format(k, rank))

    if esp is None:
        esp = elementary_symmetric_polynomials(eig_vals, k)

    mask = np.zeros(k, dtype=int)
    for n in range(eig_vals.size, 0, -1):
        if rng.rand() < eig_vals[n - 1] * esp[k - 1, n - 1] / esp[k, n]:
            k -= 1
            mask[k] = n - 1
            if k == 0:
                break

    return eig_vecs[:, mask]


def elementary_symmetric_polynomials(x, k):
    """Evaluate the `elementary symmetric polynomials <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_ :math:`[e_i(x_1, \\dots, x_m)]_{i=0, m=1}^{k, n}`.

    :param x:
        Points at which the elementary symmetric polynomials will be evaluated
    :type x:
        array_like

    :param k:
        Maximum degree of the elementary symmetric polynomials to be evaluated
    :type k:
        int

    :return:
        Matrix of size :math:`(k+1, n)` containing the evaluation of the elementary symmetric polynomials :math:`[e_i(x_1, \\dots, x_m)]_{i=0, m=1}^{k, n}`
    :rtype:
        array_like

    .. seealso::

        - :cite:`KuTa12` Algorithm 7
        - `Wikipedia <https://en.wikipedia.org/wiki/Elementary_symmetric_polynomial>`_
    """

    # Initialize output array
    n = x.size
    E = np.zeros((k + 1, n + 1), dtype=float)
    E[0, :] = 1.0

    # Recursive evaluation
    for i in range(1, k + 1):
        for m in range(0, n):
            E[i, m + 1] = E[i, m] + x[m] * E[i - 1, m]

    return E
