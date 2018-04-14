import time
import numpy as np

from cvxopt import matrix, spmatrix, solvers
solvers.options['show_progress'] = False
solvers.options['glpk'] = dict(msg_lev='GLP_MSG_OFF')

###############################################################
############## Approximate samplers for projection DPPs #######
###############################################################

def extract_basis(y_sol, eps=1e-5):
    """ Subroutine of zono_sampling to extract the tile of the zonotope 
    in which a point lies. It extracts the indices of entries of the solution 
    of LP :eq:`Px` that are in (0,1).

    :param y_sol: 
        Optimal solution of LP :eq:`Px`
    :type y_sol: 
        list
            
    :param eps: 
        Tolerance :math:`y_i^* \in (\epsilon, 1-\epsilon), \quad \epsilon \geq 0`
    :eps type: 
        float

    :return: 
        Indices of the feature vectors spanning the tile in which the point is lies. 
        :math:`B_{x} = \left\{ i \, ; \, y_i^* \in (0,1) \\right\}`
    :rtype: 
        list         
            
    .. seealso::

        Algorithm 3 in :cite:`GaBaVa17`

        - :func:`zono_sampling <zono_sampling>`
    """

    basis = np.where((eps<y_sol)&(y_sol<1-eps))[0]
    
    return basis.tolist()

def zono_sampling(Vectors, c=None, nb_it_max = 10, T_max=10):
    """ MCMC based sampler for projection DPPs.
    The similarity matrix is the orthogonal projection matrix onto 
    the row span of the feature vector matrix.
    Samples are of size equal to the rank of the projection matrix 
    also equal to the rank of the feature matrix (assumed to be full row rank).

    :param Vectors:
        Feature vector matrix, feature vectors are stacked columnwise.
        It is assumed to be full row rank.
    :type Vectors:
        array_type

    :param c: Linear objective of the linear program used to identify 
        the tile in which a point lies.
    :type c: list

    :param nb_it_max:
        Maximum number of iterations performed by the the algorithm.
        Default is 10.
    :type nb_it_max: 
        int

    :param T_max: 
        Maximum running time of the algorithm (in seconds).
        Default is 10s.
    :type T_max: 
        float

    :return:
        MCMC chain of approximate samples (stacked row_wise i.e. nb_it_max rows).
    :rtype: 
        array_type    

    .. seealso::

        Algorithm 5 in :cite:`GaBaVa17`

        - :func:`extract_basis <extract_basis>`
        - :func:`basis_exchange <basis_exchange>`
    """

    r,n = Vectors.shape

    #################################################
    ############### Linear problems #################
    #################################################
    # Canonical form
    # min 	c.T*x 			min 	c.T*x
    # s.t. 	G*x <= h 	<=>	s.t. 	G*x + s = h
    # 		A*x = b					A*x = b
    #								s >= 0
    # CVXOPT
    # =====> solvers.lp(c, G, h, A, b, solver='glpk')
    #################################################

    ### To access the tile Z(B_x) 
    # Solve P_x(A,c)
    #########################################################
    # y^* =
    # argmin	c.T*y 			argmin	c.T*y
    # s.t. 		A*y = x 	<=>	s.t. 		A 	*y 	= 	x
    # 		 	0 <= y <= 1				[ I_n]	*y 	<=	[1^n]
    #									[-I_n]			[0^n]
    #########################################################
    # Then B_x = \{ i ; y_i^* \in ]0,1[ \}
    if c is None:
        c = matrix(np.random.randn(n))

    A = spmatrix(0.0, [], [], (r, n))
    A[:,:] = Vectors

    G = spmatrix(0.0, [], [], (2*n, n))
    G[:n,:] = spmatrix(1.0, range(n), range(n))
    G[n:,:] = spmatrix(-1.0, range(n), range(n))

    ### Endpoints of segment 
    # D_x \cap Z(A) = [x+alpha_m*d, x-alpha_M*d]
    #############################################################################################
    # alpha_m/_M = 
    # argmin	+/-alpha 						argmin	[+/-1 0^n].T * [alpha,lambda]
    # s.t. 		x + alpha d = A lambda 	<=>		s.t. 	[-d A] 		 * [alpha,lambda]	=	x
    # 		 	0 <= lambda <= 1						[ 0^n I_n ]	 * [alpha,lambda]	<=	[1^n]
    #													[ 0^n -I_n]							[0^n]
    #############################################################################################

    c_mM = matrix(0.0, (n+1,1))
    c_mM[0] = 1.0

    A_mM = spmatrix(0.0, [], [], (r, n+1))
    A_mM[:,1:] = A

    G_mM = spmatrix(0.0, [], [], (2*n, n+1))
    G_mM[:,1:] = G

    # Common h to both kind of LP
    # cf. 0 <= y <= 1 and 0 <= lambda <= 1
    h = matrix(0.0, (2*n, 1))
    h[:n,:] = 1.0

    ########################
    #### Initialization ####
    ########################
    B_x0 = []
    while len(B_x0) != r:
        # Initial point x0 = A*u, u~U[0,1]^n
        x0 = matrix(Vectors.dot(np.random.rand(n)))
        # Initial tile B_x0
            # Solve P_x0(A,c)
        y_star = solvers.lp(c, G, h, A, x0, solver='glpk')['x']
            # Get the tile
        B_x0 = extract_basis(np.asarray(y_star))

    # Initialize sequence of samples
    Bases = [B_x0]

    # Compute the det of the tile (Vol(B)=abs(det(B)))
    det_B_x0 = np.linalg.det(Vectors[:,B_x0])

    it, t_start = 1, time.time()
    flag = it < nb_it_max
    while flag:

        # Take uniform direction d defining D_x0
        d = matrix(np.random.randn(r,1))

        # Define D_x0 \cap Z(A) = [x0 + alpha_m*d, x0 - alpha_M*d]
            # Update the constraint [-d A] * [alpha,lambda]	= x
        A_mM[:,0] = -d
            # Find alpha_m/M
        alpha_m = solvers.lp(c_mM, G_mM, h, A_mM, x0, solver='glpk')['x'][0]
        alpha_M = solvers.lp(-c_mM, G_mM, h, A_mM, x0, solver='glpk')['x'][0]

        # Propose x1 ~ U_{[x0+alpha_m*d, x0-alpha_M*d]}
        x1 = x0 + (alpha_m + (alpha_M-alpha_m)*np.random.rand())*d
        # Proposed tile B_x1
            # Solve P_x1(A,c)
        y_star = solvers.lp(c, G, h, A, x1, solver='glpk')['x']
            # Get the tile
        B_x1 = extract_basis(np.asarray(y_star))

        # Accept/Reject the move with proba Vol(B1)/Vol(B0)
        if len(B_x1) != r: # In case extract_basis returned smtg ill conditioned
            Bases.append(B_x0)
        else:
            det_B_x1 = np.linalg.det(Vectors[:,B_x1])
            if np.random.rand() < abs(det_B_x1/det_B_x0):
                x0, B_x0, det_B_x0 = x1, B_x1, det_B_x1
                Bases.append(B_x1)
            else:
                Bases.append(B_x0)

        if nb_it_max is not None:
            it += 1
            flag = it < nb_it_max
        else:
            flag = time.time() - t_start < T_max

    print("Time enlapsed", time.time()-t_start)

    return np.array(Bases)