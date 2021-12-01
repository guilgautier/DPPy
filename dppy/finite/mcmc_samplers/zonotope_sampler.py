import numpy as np
import scipy.linalg as la
from cvxopt import matrix, solvers, spmatrix

from dppy.utils import check_random_state


def zonotope_sampler(dpp, random_state=None, **params):
    if not dpp.projection or not dpp.hermitian or dpp.A_zono is None:
        raise ValueError(
            "DPP must be defined via FiniteDPP(kernel_type='correlation', projection=True, hermitian=True, A_zono=...)"
        )
    return zonotope_sampler_core(dpp.A_zono, random_state, **params)


def zonotope_sampler_core(
    A_zono,
    random_state=None,
    lin_obj=None,
    x0=None,
    max_iter=10,
    show_progress=False,
    msg_lev="GLP_MSG_OFF",
):
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
        - ``'max_iter'`` (int): Number of iterations of the MCMC chain. Default is 10.
        - ``'T_max'`` (float): Maximum running time of the algorithm (in seconds).
        Default is None.
        - ``'random_state`` (default None)
    :type params: dict

    :return:
        MCMC chain of approximate samples (stacked row_wise i.e. max_iter rows).
    :rtype:
        list of lists

    .. seealso::

        - :cite:`GaBaVa17` Algorithm 5
        - :func:`extract_basis <extract_basis>`
        - :func:`exchange_sampler <exchange_sampler>`
    """
    solvers.options["show_progress"] = show_progress
    solvers.options["glpk"] = {"msg_lev": msg_lev}

    rng = check_random_state(random_state)

    r, N = A_zono.shape  # sample size r = rank(A_zono), ground set size N
    # Linear objective
    if lin_obj is None:
        lin_obj = rng.randn(N)
    c = matrix(lin_obj)

    # Initial point x0 = A*u, u~U[0,1]^n
    if x0 is None:
        x0 = A_zono.dot(rng.rand(N))
    x0 = matrix(x0)

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
    chain = np.zeros((max_iter, r), dtype=int)
    chain[0] = B_x0

    # Compute the det of the tile (Vol(B)=abs(det(B)))
    det_B_x0 = la.det(A_zono[:, B_x0])

    for it in range(1, max_iter):

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

    return chain.tolist()


def extract_basis(y_sol, tol=1e-5):
    r"""Extract the basis/tile :math:`B_{x}` of the zonotope in which the RHS :math:`x` of the LP :eq:`eq:Px`, given its optimal solution ``y_sol``.

    .. math::

        B_{x} = \left\{ i \, ; \, y_i^* \in (0,1) \right\}

    :param y_sol:
        Optimal solution of the LP :eq:`eq:Px`
    :type y_sol:
        array_like

    :param tol:
        Tolerance :math:`y_i^* \in (\epsilon, 1-\epsilon), \quad \epsilon \geq 0`
    :type eps:
        float

    :return:
        Indices of the feature vectors spanning the tile in which the point is lies.
    :rtype:
        array_like

    .. seealso::

        - :cite:`GaBaVa17` Algorithm 3
        - :func:`zonotope_sampler_core <zonotope_sampler_core>`
    """
    mask = tol < y_sol
    np.logical_and(mask, y_sol < 1 - tol, where=mask, out=mask)
    return np.where(mask)[0]
