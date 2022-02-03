from warnings import warn

import numpy as np
import scipy.linalg as la

from dppy.finite.exact_samplers.projection_sampler_eigen import (
    select_projection_sampler_eigen,
)
from dppy.utils import check_random_state, elementary_symmetric_polynomials


def spectral_sampler_dpp(dpp, random_state=None, **params):
    r"""Generate an exact sample from an hermitian ``dpp`` using the :ref:`spectral method <finite_dpps_exact_sampling_spectral_method>`.

    The eigenvalues ``dpp.K_eig_vals`` and eigenvectors ``dpp.eig_vecs`` of the correlation kernel :math:`\mathbf{K}` are computed from the current parametrization of ``dpp``.

    :param dpp:
        Finite DPP
    :type dpp:
        :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param random_state:
        random number generator or seed, defaults to None
    :type random_state:
        optional

    Keyword arguments:
        - **mode** (str): select the variant of the sampler used in the second step which boils down to sampling from a projection DPP, see :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.select_projection_sampler_eigen`

    :return: sample
    :rtype: list
    """
    assert dpp.hermitian
    compute_spectral_sampler_parameters_dpp(dpp)
    return do_spectral_sampler_dpp(dpp, random_state, **params)


def do_spectral_sampler_dpp(dpp, random_state=None, **params):
    """Perform the main steps of the :ref:`spectral method <finite_dpps_exact_sampling_spectral_method>` to generate an exact sample from ``dpp``.

    :param dpp: Finite DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param random_state: random number generator or seed, defaults to None
    :type random_state: optional

    :return: sample
    :rtype: list
    """
    rng = check_random_state(random_state)
    eig_vals, eig_vecs = dpp.K_eig_vals, dpp.eig_vecs
    V = select_eigen_vectors_dpp(eig_vals, eig_vecs, random_state=rng)
    mode = params.get("mode", "")
    sampler = select_projection_sampler_eigen(mode)
    return sampler(V, random_state=rng)


def compute_spectral_sampler_parameters_dpp(dpp):
    r"""Compute eigenvalues and eigenvectors of correlation kernel :math:`\mathbf{K}` from the current parametrization of ``dpp``. These values are stored in the ``dpp.K_eig_vals`` and ``dpp.eig_vecs`` attributes.

    :param dpp: Finite hermitian DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`
    """
    while compute_spectral_sampler_parameters_dpp_step(dpp):
        pass


def compute_spectral_sampler_parameters_dpp_step(dpp):
    r"""Compute eigenvalues and eigenvectors of correlation kernel :math:`\mathbf{K}` from the current parametrization of ``dpp``. These values are stored in the ``dpp.K_eig_vals`` and ``dpp.eig_vecs`` attributes.

    This corresponds to a sort of fixed point algorithm to compute eigenvalues and eigenvectors.

    :return: ``False`` if the right parameters are indeed computed, ``True`` if extra computations are required.
    :rtype: bool
    """
    if dpp.K_eig_vals is not None:
        return False

    if dpp.L_eig_vals is not None:
        dpp.K_eig_vals = dpp.L_eig_vals / (1.0 + dpp.L_eig_vals)
        return False

    if dpp.K is not None:  # 0 <= K <= I
        eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
        np.clip(eig_vals, 0.0, 1.0, out=eig_vals)
        dpp.K_eig_vals = eig_vals
        return False

    if dpp.L_gram_factor is not None:
        # L_dual = Phi * Phi.T = W Theta W.T
        # L = Phi.T Phi = V Gamma V
        # then Gamma = Theta and V = Phi.T W Theta^{-1/2}
        Phi = dpp.L_gram_factor
        d, N = Phi.shape
        if d >= N:
            dpp.L = Phi.T.dot(Phi)
            return True

        L_dual = Phi.dot(Phi.T)
        eig_vals, eig_vecs = la.eigh(L_dual)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        dpp.eig_vecs = Phi.T.dot(eig_vecs / np.sqrt(eig_vals))
        return True

    if dpp.L is not None:
        eig_vals, dpp.eig_vecs = la.eigh(dpp.L)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        return True

    if dpp.A_zono is not None:  # K = A.T (A A.T)^-1 A (orthogonal projection)
        A = dpp.A_zono
        dpp.K_eig_vals = np.ones(len(A), dtype=float)
        dpp.eig_vecs, *_ = la.qr(A.T, mode="economic")
        return False

    if dpp.eval_L is not None and dpp.X_data is not None:
        dpp.compute_L()
        return True

    raise ValueError(
        "Failed to compute spectral sampler parameters (K eigenvalues and eigenvectors). This should never happen, please consider rasing an issue on github at https://github.com/guilgautier/DPPy/issues"
    )


# Phase 1
def select_eigen_vectors_dpp(bernoulli_params, eig_vecs, random_state=None):
    """Select columns of ``eig_vecs`` by sampling Bernoulli variables with parameters ``bernoulli_params``.

    :param bernoulli_params:
        Parameters of Bernoulli variables
    :type bernoulli_params:
        array_like, shape (r,)

    :param eig_vecs:
        Eigenvectors, stored as columns of a 2d array
    :type eig_vecs:
        array_like, shape (N, r)

    :return:
        Selected eigenvectors
    :rtype:
        array_like

    .. seealso::

        - :func:`dpp_sampler_eig <dpp_sampler_eig>`
    """
    rng = check_random_state(random_state)
    mask = rng.rand(bernoulli_params.size) < bernoulli_params
    return eig_vecs[:, mask]


#### k-DPP


def spectral_sampler_k_dpp(dpp, size, random_state=None, **params):
    r"""Generate an exact sample from an hermitian :math:`k\!\operatorname{-DPP}` associated with ``dpp`` and :math:`k=` ``size``, using the :ref:`spectral method <finite_dpps_exact_sampling_k_dpps>`.

    The precomputation cost of generating the first sample involves computing the eigenvalues and eigenvectors of the likelihood kernel :math:`\mathbf{L}` from the current parametrization of ``dpp`` and stored in the ``dpp.L_eig_vals`` and ``dpp.eig_vecs`` attributes.
    The elementary symmetric polynomials of degree :math:`0` to :math:`k` are also evaluated at the eigenvalues of the likelihood kernel and stored in the ``dpp.esp`` attribute, see :py:func:`~dppy.utils.elementary_symmetric_polynomials`.

    :param dpp:
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param size: size :math:`k` of the output sample
    :type size: int

    :param random_state: random number generator or seed, defaults to None
    :type random_state: optional

    Keyword arguments:
        - **mode** (str): select the variant of the sampler used in the second step which boils down to sampling from a projection DPP, see :py:func:`~dppy.finite.exact_samplers.projection_sampler_eigen.select_projection_sampler_eigen`

    :return: sample
    :rtype: list
    """
    assert dpp.hermitian
    if not dpp.projection:
        compute_spectral_sampler_parameters_k_dpp(dpp, size)
        return do_spectral_sampler_k_dpp(dpp, size, random_state, **params)
    else:
        eig_vals = compute_spectral_sampler_eig_vals_projection_k_dpp(dpp, size)
        # Phase 1 select_eigenvectors from eigvalues = 0 or 1
        V = dpp.eig_vecs[:, eig_vals > 0.5]
        # Phase 2
        dpp.size_k_dpp = size
        sampler = select_projection_sampler_eigen(params.get("mode"))
        return sampler(V, size=size, random_state=random_state)


def do_spectral_sampler_k_dpp(dpp, size, random_state=None, **params):
    r"""Perform the main steps of the :ref:`spectral method <finite_dpps_exact_sampling_k_dpps>` to generate an exact sample from the :math:`k\!\operatorname{-DPP}` associated with ``dpp`` and :math:`k=` ``size``.

    :param dpp:
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param size: size :math:`k` of the output sample
    :type size: int

    :param random_state: random number generator or seed, defaults to None
    :type random_state: optional

    :return: sample
    :rtype: list
    """
    rng = check_random_state(random_state)
    # Phase 1
    eig_vals, eig_vecs = dpp.L_eig_vals, dpp.eig_vecs
    V = select_eigen_vectors_k_dpp(
        eig_vals,
        eig_vecs,
        size,
        esp=dpp.esp,
        random_state=rng,
    )
    # Phase 2
    dpp.size_k_dpp = size
    sampler = select_projection_sampler_eigen(params.get("mode"))
    return sampler(V, size=size, random_state=rng)


def compute_spectral_sampler_parameters_k_dpp(dpp, size):
    r"""Compute the eigenvalues and eigenvectors of the likelihood kernel :math:`\mathbf{L}` the current parametrization of ``dpp`` and stored in the ``dpp.L_eig_vals`` and ``dpp.eig_vecs`` attributes.
    The elementary symmetric polynomials of degree :math:`0` to :math:`k` are also evaluated at the eigenvalues of the likelihood kernel and stored in the ``dpp.esp`` attribute, see :py:func:`~dppy.utils.elementary_symmetric_polynomials`.

    :param dpp: Finite hermitian DPP
    :type dpp: FiniteDPP

    :param size: size :math:`k` of the output sample
    :type size: int
    """
    while compute_spectral_sampler_parameters_k_dpp_step(dpp, size):
        pass


def compute_spectral_sampler_parameters_k_dpp_step(dpp, size):
    r"""Compute eigenvalues and eigenvectors of correlation kernel :math:`\mathbf{L}` from the current parametrization of ``dpp``. These values are stored in the ``dpp.L_eig_vals`` and ``dpp.eig_vecs`` attributes. The elementary symmetric polynomials of degree :math:`0` to :math:`k` are also evaluated at the eigenvalues of the likelihood kernel and stored in the ``dpp.esp`` attribute, see :py:func:`~dppy.utils.elementary_symmetric_polynomials`.

    This corresponds to a sort of fixed point algorithm to compute eigenvalues and eigenvectors.

    :return: ``False`` if the right parameters are indeed computed, ``True`` if extra computations are required.
    :rtype: bool
    """

    if dpp.L_eig_vals is not None:
        # Phase 1
        # Precompute elementary symmetric polynomials
        if not dpp.projection:
            if dpp.esp is None or dpp.size_k_dpp < size:
                dpp.esp = elementary_symmetric_polynomials(dpp.L_eig_vals, size)
        return False

    elif dpp.K_eig_vals is not None:
        np.seterr(divide="raise")
        dpp.L_eig_vals = dpp.K_eig_vals / (1.0 - dpp.K_eig_vals)
        return True

    if dpp.L_gram_factor is not None:
        # L_dual = Phi * Phi.T = W Theta W.T
        # L = Phi.T Phi = V Gamma V
        # then Gamma = Theta and V = Phi.T W Theta^{-1/2}
        Phi = dpp.L_gram_factor
        d, N = Phi.shape
        if d >= N:
            dpp.L = Phi.T.dot(Phi)
            return True

        L_dual = Phi.dot(Phi.T)
        eig_vals, eig_vecs = la.eigh(L_dual)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        dpp.eig_vecs = Phi.T.dot(eig_vecs / np.sqrt(eig_vals))
        return True

    elif dpp.L is not None:
        eig_vals, dpp.eig_vecs = la.eigh(dpp.L)
        np.fmax(eig_vals, 0.0, out=eig_vals)
        dpp.L_eig_vals = eig_vals
        return True

    elif dpp.K is not None:
        eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
        np.clip(eig_vals, 0.0, 1.0, out=eig_vals)
        dpp.K_eig_vals = eig_vals
        return True

    elif dpp.eval_L is not None and dpp.X_data is not None:
        # In case mode!="vfx"
        dpp.compute_L()
        return True

    else:
        raise ValueError(
            "None of the available samplers could be used based on the current DPP representation. This should never happen, please consider rasing an issue on github at https://github.com/guilgautier/DPPy/issues"
        )


def select_eigen_vectors_k_dpp(eig_vals, eig_vecs, size, esp=None, random_state=None):
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
        Computation of the elementary symmetric polynomials previously evaluated in ``eig_vals`` and returned by :py:func:`~dppy.utils.elementary_symmetric_polynomials`, default to None.
    :type esp:
        array_like

    :return:
        Selected eigenvectors
    :rtype:
        array_like

    .. seealso::

        - :cite:`KuTa12` Algorithm 8
        - :py:func:`~dppy.utils.elementary_symmetric_polynomials`
    """

    rng = check_random_state(random_state)

    # Size of: ground set / sample
    N, k = eig_vecs.shape[0], size

    # as in np.linalg.matrix_rank
    tol = np.max(eig_vals) * N * np.finfo(float).eps
    rank = np.count_nonzero(eig_vals > tol)
    if k > rank:
        raise ValueError("size k={} > rank(L)={}".format(k, rank))

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


def compute_spectral_sampler_eig_vals_projection_k_dpp(dpp, size):
    r"""Compute the eigenvalues of the projection kernel :math:`\mathbf{L}` or :math:`\mathbf{K}` according to the attribute ``dpp.kernel_type``.

    :param dpp: Finite DPP
    :type dpp: :py:class:`~dppy.finite.dpp.FiniteDPP`

    :param size: size :math:`k` of the output sample
    :type size: int

    :raises ValueError: If ``dpp`` is a projection :math:`\operatorname{DPP}(\mathbf{K})` and ``size`` :math:`\neq \operatorname{rank}(\mathbf{K})`.

    :return: Vector of eigenvalues.
    :rtype: array_like
    """
    assert dpp.projection
    if dpp.kernel_type == "likelihood":
        compute_spectral_sampler_parameters_k_dpp(dpp, size)
        return dpp.L_eig_vals
    if dpp.kernel_type == "correlation":
        # check size = rank(K)
        if dpp.K_eig_vals is not None:
            rank = np.rint(np.sum(dpp.K_eig_vals)).astype(int)
        elif dpp.A_zono is not None:
            rank = dpp.A_zono.shape[0]
        else:
            dpp.compute_K()
            rank = np.rint(np.trace(dpp.K)).astype(int)

        if size != rank:
            raise ValueError(
                "k-DPP(K) with projection correlation kernel is only defined for k = rank(K), here k={} != rank={}".format(
                    size, rank
                )
            )

        if dpp.K_eig_vals is not None:
            return dpp.K_eig_vals
        if dpp.A_zono is not None:
            warn(
                "DPP defined via `A_zono`, apriori you want to use `sampl_mcmc`, but you have called `sample_exact`"
            )
            dpp.K_eig_vals = np.ones(rank)
            dpp.eig_vecs, *_ = la.qr(dpp.A_zono.T, mode="economic")
            return dpp.K_eig_vals
        else:
            dpp.compute_K()  # 0 <= K <= I
            eig_vals, dpp.eig_vecs = la.eigh(dpp.K)
            np.clip(eig_vals, 0.0, 1.0, out=eig_vals)
            dpp.K_eig_vals = eig_vals
            return dpp.K_eig_vals
