from dppy.finite.dpp import FiniteDPP
from dppy.utils import (
    is_equal_to_O_or_1,
    is_orthonormal_columns,
    is_projection,
    is_symmetric,
)

VALID_DPP_KERNEL_PARAMS = {
    "correlation": {
        "K": {
            "projection": (True, False),
            "hermitian": (True, False),
            "expression": "K",
            "description": "0 <= K (N, N) <= I if hermitian=True",
        },
        "K_eig_dec": {
            "projection": (True, False),
            "hermitian": (True,),
            "expression": "(e_vals, e_vecs)",
            "description": "0 <= e_vals (r,) <= 1, e_vecs (N, r)",
        },
        "A_zono": {
            "projection": (True,),
            "hermitian": (True,),
            "expression": "A",
            "description": "A (d, N) defines projection K = A.T (A A.T)^-1 A",
        },
    },
    "likelihood": {
        "L": {
            "projection": (True, False),
            "hermitian": (True,),
            "expresion": "L",
            "description": "L (N, N) >= 0 if hermitian=True",
        },
        "L_eig_dec": {
            "projection": (True, False),
            "hermitian": (True,),
            "expression": "(e_vals, e_vecs)",
            "description": "e_vals (r,) >= 0, e_vecs (N, r)",
        },
        "L_gram_factor": {
            "projection": (False,),
            "hermitian": (True,),
            "expression": "Phi",
            "description": "Phi (d, N) with L = Phi.T Phi",
        },
        "L_eval_X_data": {
            "projection": (False,),
            "hermitian": (True,),
            "expression": "(eval_L, X_data)",
            "description": "X_data (d, N), eval_L callable positive semi-definite kernel function",
        },
    },
}


def check_arguments_coherence(kernel_type, projection, hermitian, **params):
    """Check coherence of initialization parameters of :py:class:`~dppy.finite.dpp.FiniteDPP`"""

    valid_kernel_types = VALID_DPP_KERNEL_PARAMS.keys()
    if kernel_type not in valid_kernel_types:
        raise ValueError(f"kernel_type not in {valid_kernel_types}")

    valid_params = VALID_DPP_KERNEL_PARAMS[kernel_type]
    coherent_params = set(params).intersection(valid_params)
    if not coherent_params:
        suggestions = []
        for param, _dict in valid_params.items():
            proj, herm, expr, descr = _dict.values()
            args = ", ".join(
                [
                    f'kernel_type="{kernel_type}"',
                    f"projection={'?'.join(map(str, proj))}",
                    f"hermitian={'?'.join(map(str, herm))}",
                    f"{param}={expr}",
                ]
            )
            suggestions.append(f"- FiniteDPP({args})\nwhere {descr}")
        suggestions = "\n".join(suggestions)
        raise ValueError(f"Invalid DPP parametrization choose:\n{suggestions}")

    for param in coherent_params:
        proj, herm, expr, descr = valid_params[param].values()
        s = f"for FiniteDPP(..., {param}={expr}), where {descr}"
        if projection not in proj:
            _proj = "?".join(map(str, proj))
            raise ValueError(f"Argument projection != {_proj} {s}")
        if hermitian not in herm:
            _herm = "?".join(map(str, herm))
            raise ValueError(f"Argument hermitian != {_herm} {s}")


def check_parameters_validity(dpp):
    # Attributes relative to K correlation kernel:
    # K, K_eig_vals, K_eig_vecs, A_zono

    if dpp.kernel_type == "correlation":
        if dpp.projection:
            is_equal_to_O_or_1(dpp.K_eig_vals)
            is_projection(dpp.K)
        if dpp.hermitian:
            is_symmetric(dpp.K)
            is_orthonormal_columns(dpp.eig_vecs)

    if dpp.kernel_type == "likelihood":
        if dpp.projection:
            is_equal_to_O_or_1(dpp.L_eig_vals)
            is_projection(dpp.L)
        if dpp.hermitian:
            is_symmetric(dpp.L)
            is_orthonormal_columns(dpp.eig_vecs)

    if dpp.eval_L is not None:
        assert callable(dpp.eval_L)
    if dpp.X_data is not None:
        assert dpp.X_data.ndim == 2 and dpp.X_data.size


def ground_set_size(dpp):
    assert isinstance(dpp, FiniteDPP)
    if dpp.K is not None:
        return dpp.K.shape[0]
    if dpp.L is not None:
        return dpp.L.shape[0]
    if dpp.L_gram_factor is not None:
        return dpp.L_gram_factor.shape[1]
    if dpp.eig_vecs is not None:
        return dpp.eig_vecs.shape[0]
    if dpp.X_data is not None:
        return dpp.X_data.shape[0]
    if dpp.A_zono is not None:
        return dpp.A_zono.shape[1]
    raise ValueError(
        "The size of the ground set of the finite 'dpp' cannot be computed from the current parametrization"
    )
