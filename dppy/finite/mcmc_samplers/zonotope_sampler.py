import time

import numpy as np
import scipy.linalg as la

from dppy.utils import check_random_state, det_ST


def zonotope_sampler(A_zono, **params):
    """MCMC based sampler for projection DPPs.
    The similarity matrix is the orthogonal projection matrix onto
    the row span of the feature vector matrix.
    Samples are of size equal to the ransampl_size of the projection matrix
    also equal to the rank of the feature matrix (assumed to be full row rank).

    :param A_zono:
        Feature vector matrix, feature vectors are stacked columnwise.
        It is assumed to be full row rank.
    :type A_zono:
        array_like

    :param params: Dictionary containing the parameters

        - ``'lin_obj'`` (list): Linear objective (:math:`c`) of the linear program used to identify the tile in which a point lies. Default is a random Gaussian vector.
        - ``'x_0'` (list): Initial point.
        - ``'nb_iter'`` (int): Number of iterations of the MCMC chain. Default is 10.
        - ``'T_max'`` (float): Maximum running time of the algorithm (in seconds).
        Default is None.
        - ``'random_state`` (default None)
    :type params: dict

    :return:
        MCMC chain of approximate samples (stacked row_wise i.e. nb_iter rows).
    :rtype:
        list of lists

    .. seealso::

        Algorithm 5 in :cite:`GaBaVa17`

        - :func:`extract_basis <extract_basis>`
        - :func:`exchange_sampler <exchange_sampler>`
    """
    # For zonotope sampler
    try:
        from cvxopt import matrix, solvers, spmatrix
    except ImportError:
        raise ValueError(
            "The cvxopt package is required to use the zonotype sampler (see setup.py)."
        )

    solvers.options["show_progress"] = params.get("show_progress", False)
    solvers.options["glpk"] = {"msg_lev": params.get("show_progress", "GLP_MSG_OFF")}

    rng = check_random_state(params.get("random_state", None))

    r, N = A_zono.shape  # Sizes of r=samples=rank(A_zono), N=ground set
    # Linear objective
    c = matrix(params.get("lin_obj", rng.randn(N)))
    # Initial point x0 = A*u, u~U[0,1]^n
    x0 = matrix(params.get("x_0", A_zono.dot(rng.rand(N))))

    nb_iter = params.get("nb_iter", 10)
    T_max = params.get("T_max", None)

    ###################
    # Linear problems #
    ###################
    # Canonical form
    # min       c.T*x         min     c.T*x
    # s.t.  G*x <= h    <=>   s.t.    G*x + s = h
    #        A*x = b                      A*x = b
    #                                      s >= 0
    # CVXOPT
    # =====> solvers.lp(c, G, h, A, b, solver='glpk')
    #################################################

    # To access the tile Z(B_x)
    # Solve P_x(A,c)
    ######################################################
    # y^* =
    # argmin  c.T*y               argmin  c.T*y
    # s.t.  A*y = x         <=>   s.t.  A  *y  = x
    #       0 <= y <= 1             [ I_n] *y <= [1^n]
    #                               [-I_n]       [0^n]
    ######################################################
    # Then B_x = \{ i ; y_i^* \in ]0,1[ \}

    A = spmatrix(0.0, [], [], (r, N))
    A[:, :] = A_zono

    G = spmatrix(0.0, [], [], (2 * N, N))
    G[:N, :] = spmatrix(1.0, range(N), range(N))
    G[N:, :] = spmatrix(-1.0, range(N), range(N))

    # Endpoints of segment
    # D_x \cap Z(A) = [x+alpha_m*d, x-alpha_M*d]
    ###########################################################################
    # alpha_m/_M = argmin  +/-alpha      argmin [+/-1 0^N].T * [alpha,lambda]
    # s.t.    x + alpha d = A lambda <=> s.t.  [-d A] *[alpha, lambda] = x
    #         0 <= lambda <= 1             [0^N I_N] *[alpha, lambda] <= [1^N]
    #                                      [0^N -I_N]                    [0^N]
    ##########################################################################

    c_mM = matrix(0.0, (N + 1, 1))
    c_mM[0] = 1.0

    A_mM = spmatrix(0.0, [], [], (r, N + 1))
    A_mM[:, 1:] = A

    G_mM = spmatrix(0.0, [], [], (2 * N, N + 1))
    G_mM[:, 1:] = G

    # Common h to both kind of LP
    # cf. 0 <= y <= 1 and 0 <= lambda <= 1
    h = matrix(0.0, (2 * N, 1))
    h[:N, :] = 1.0

    ##################
    # Initialization #
    ##################
    B_x0 = []
    while len(B_x0) != r:
        # Initial tile B_x0
        # Solve P_x0(A,c)
        y_star = solvers.lp(c, G, h, A, x0, solver="glpk")["x"]
        # Get the tile
        B_x0 = extract_basis(np.asarray(y_star))

    # Initialize sequence of sample
    chain = np.zeros((nb_iter, r), dtype=int)
    chain[0] = B_x0

    # Compute the det of the tile (Vol(B)=abs(det(B)))
    det_B_x0 = la.det(A_zono[:, B_x0])

    t_start = time.time() if T_max else 0

    for it in range(1, nb_iter):

        # Take uniform direction d defining D_x0
        d = matrix(rng.randn(r, 1))

        # Define D_x0 \cap Z(A) = [x0 + alpha_m*d, x0 - alpha_M*d]
        # Update the constraint [-d A] * [alpha,lambda] = x
        A_mM[:, 0] = -d
        # Find alpha_m/M
        alpha_m = solvers.lp(c_mM, G_mM, h, A_mM, x0, solver="glpk")["x"][0]
        alpha_M = solvers.lp(-c_mM, G_mM, h, A_mM, x0, solver="glpk")["x"][0]

        # Propose x1 ~ U_{[x0+alpha_m*d, x0-alpha_M*d]}
        x1 = x0 + (alpha_m + (alpha_M - alpha_m) * rng.rand()) * d
        # Proposed tile B_x1
        # Solve P_x1(A,c)
        y_star = solvers.lp(c, G, h, A, x1, solver="glpk")["x"]
        # Get the tile
        B_x1 = extract_basis(np.asarray(y_star))

        # Accept/Reject the move with proba Vol(B1)/Vol(B0)
        if len(B_x1) != r:  # if extract_basis returned smtg ill conditioned
            chain[it] = B_x0
        else:
            det_B_x1 = la.det(A_zono[:, B_x1])
            if rng.rand() < abs(det_B_x1 / det_B_x0):
                x0, B_x0, det_B_x0 = x1, B_x1, det_B_x1
                chain[it] = B_x1
            else:
                chain[it] = B_x0

        if T_max and time.time() - t_start < T_max:
            break

    return chain.tolist()


def extract_basis(y_sol, eps=1e-5):
    """Subroutine of :func:`zonotope_sampler <zonotope_sampler>` to extract the tile of the zonotope
    in which a point lies. It extracts the indices of entries of the solution
    of LP :eq:`eq:Px` that are in (0,1).

    :param y_sol:
        Optimal solution of LP :eq:`eq:Px`
    :type y_sol:
        list

    :param eps:
        Tolerance :math:`y_i^* \\in (\\epsilon, 1-\\epsilon), \\quad \\epsilon \\geq 0`
    :eps type:
        float

    :return:
        Indices of the feature vectors spanning the tile in which the point is lies.
        :math:`B_{x} = \\left\\{ i \\, ; \\, y_i^* \\in (0,1) \\right\\}`
    :rtype:
        list

    .. seealso::

        Algorithm 3 in :cite:`GaBaVa17`

        - :func:`zonotope_sampler <zonotope_sampler>`
    """

    basis = np.where((eps < y_sol) & (y_sol < 1 - eps))[0]

    return basis
