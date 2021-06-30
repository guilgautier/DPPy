import numpy as np
from ..utils import check_random_state, get_progress_bar
from .intermediate_sampling import (
    alpha_dpp_sampling_precompute_constants,
    alpha_dpp_sampling_do_sampling_loop,
)


def alpha_sampler(dpp, rng, **params):
    if dpp.eval_L is None or dpp.X_data is None:
        raise ValueError(
            "The alpha sampler is currently only available with "
            '{"L_eval_X_data": (L_eval, X_data)} representation.'
        )

    r_state_outer = None
    if "random_state" in params:
        r_state_outer = params.pop("random_state", None)

    sample, dpp.intermediate_sample_info = alpha_dpp_sampler(
        dpp.intermediate_sample_info, dpp.X_data, dpp.eval_L, random_state=rng, **params
    )
    if r_state_outer:
        params["random_state"] = r_state_outer

    return sample


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
