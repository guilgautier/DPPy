import numpy as np
import scipy.linalg as la

from dppy.utils import det_ST, check_random_state


def bound_min_max_eigvals(A):
    """Assuming A \\succeq 0, returns coarse lower/upper bound on the smallest/largest eigenvalue of A"""

    radius = np.sum(np.abs(A), axis=1)

    # gershgorin
    low_bnd_eig_A_min = max(1e-8, min(2 * np.diagonal(A) - radius))
    # one-norm(A)
    upp_bnd_eig_A_max = max(radius)

    return low_bnd_eig_A_min, upp_bnd_eig_A_max


def lower_upper_bounds_bif_iterator(A, x, eig_A_min, eig_A_max):
    """Compute lower and upper bounds on the bilinear inverse form :math:`x^{\top} A^{-1} x` using Gauss quadrature Lanczos.
    """

    beta = 0.0
    y_old, y = 1.0, 1.0
    c = 1.0

    b, b_u, b_l, b_t = 0.0, 0.0, 0.0, 0.0
    d, d_u, d_l = 1.0, 0.0, 0.0
    eta, eta_t = 1.0, 0.0
    w, w_u, w_l, w_t = 0.0, eig_A_min, eig_A_max, 0.0

    norm2_x = x.dot(x)

    u = x.copy()
    norm2_u_old, norm2_u = norm2_x, 1.0

    p = u.copy()
    A_dot_p = np.dot(A, p)

    while True:

        y = norm2_u_old / p.dot(A_dot_p)
        w = 1.0 / y + beta / y_old
        y_old = y

        u -= y * A_dot_p
        norm2_u = u.dot(u)

        beta = norm2_u / norm2_u_old
        norm2_u_old = norm2_u

        p[:] = u + beta * p
        A_dot_p[:] = np.dot(A, p)

        c *= eta / d**2

        d = 1.0 / y
        d_u = w - w_u
        d_l = w - w_l

        eta = beta / y**2

        w_l = eig_A_max + eta / d_l
        w_u = eig_A_min + eta / d_u
        w_t = d_u * d_l / (d_l - d_u)

        eta_t = w_t * (eig_A_max - eig_A_min)
        w_t *= eig_A_max / d_u - eig_A_min / d_l

        b += y * c
        b_l = b + eta * c / (d * (w_l * d - eta))
        b_u = b + eta * c / (d * (w_u * d - eta))
        b_t = b + eta_t * c / (d * (w_t * d - eta_t))

        lower_bound = norm2_x * max(b, b_l)
        upper_bound = norm2_x * min(b_u, b_t)

        yield lower_bound, upper_bound

        if eta < 1e-10 or np.sqrt(norm2_u) < 1e-10:
            break


def judge_exchange_gauss_quadrature(unif, kernel, sample, x_del, y_add):
    """Check whether

    .. math::

        u \\leq \\frac{\\det L_{S-x+y}}{\\det L_S}
        \\Longleftrightarrow
        u \\leq \\frac{L_{yy} - L_{y, S-x} L_{S-x}^{-1} L_{S-x, y}}
                      {L_{xx} - L_{x, S-x} L_{S-x}^{-1} L_{S-x, x}}
        \\Longleftrightarrow
        u L_{xx} - L_{yy} \\leq
            p L_{x, S-x} L_{S-x}^{-1} L_{S-x, x}
            - L_{y, S-x} L_{S-x}^{-1} L_{S-x, y}

    by computing upper and lower bounds on the two bilinear inverse terms obtained via gaussian quadrature.
    """

    S = sample.copy()
    S.remove(x_del)

    L_SS = kernel[np.ix_(S, S)]
    L_Sx, L_Sy = kernel[S, x_del], kernel[S, y_add]
    e_min, e_max = bound_min_max_eigvals(L_SS)

    iter_x = lower_upper_bounds_bif_iterator(L_SS, L_Sx, e_min, e_max)
    iter_y = lower_upper_bounds_bif_iterator(L_SS, L_Sy, e_min, e_max)

    lw_bnd_x, up_bnd_x = next(iter_x)
    lw_bnd_y, up_bnd_y = next(iter_y)

    thresh = unif * kernel[x_del, x_del] - kernel[y_add, y_add]

    while True:  # refine upper and lower bounds

        gap_x = up_bnd_x - lw_bnd_x
        gap_y = up_bnd_y - lw_bnd_y

        # choose the term which must be updated first
        if unif * gap_x - gap_y < 0:
            lw_bnd_y, up_bnd_y = next(iter_y)
        else:
            lw_bnd_x, up_bnd_x = next(iter_x)

        if thresh <= unif * lw_bnd_x - up_bnd_y:
            return True
        elif thresh >= unif * up_bnd_x - lw_bnd_y:
            return False
