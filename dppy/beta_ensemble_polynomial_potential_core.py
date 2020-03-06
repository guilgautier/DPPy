import numpy as np

from scipy.optimize import newton, brentq
from scipy.special import logsumexp, airy

from dppy.utils import check_random_state


def shift_pol(P, x0):
    """ Return polynomial P(x-x0) as poly1d object
    """

    X_x0 = np.poly1d([1, -x0])

    return sum(X_x0**n * c_n for n, c_n in enumerate(P.coeffs[::-1]))


def log_pdf_convex_quartic(x, P):
    """Evaluate :math:`\\ln` of

    .. math::

        \\pi(x) \\propto \\exp^{-(\\alpha x^4 + \\beta x^2 + \\gamma x)}
    """

    return -P(x)


def find_mode_convex_quartic(P):
    """ Compute mode of :math:`\\pi \\propto \\exp^{-P(x)}`,
    with :math:`P` convex polynomial.
    """

    roots = P.deriv(m=1).roots

    return roots[np.isreal(roots)].real[0]


def find_b_a_convex_quartic(P, mode):
    """ Compute :math:`b < 0 < a` solutions of :math:`\\pi(m+x)=\\frac{\\pi(m)}{4}`
    where :math:`\\pi \\propto \\exp^{-P(x)}` with :math:`P` convex polynomial.

    .. math::

        \\pi(m + x) = \\frac{\\pi(m)}{4}
        \\Longleftrightarrow
            P(m + x) - P(m) - 2\\ln(2) = 0
    """

    roots = (shift_pol(P, -mode) - P(mode) - 2.0 * np.log(2.0)).roots

    return np.sort(roots[np.isreal(roots)].real)


def log_pdf_convex_gen_gamma(x, shape, P):
    """Evaluate :math:`\\ln` of

    .. math::

        \\pi(x) \\propto x^{\\alpha-1} \\exp^{-(\\beta x^2 + \\gamma x)}
    """

    return (shape - 1) * np.log(x) - P(x)


def find_mode_convex_gen_gamma(shape, P):
    """ Compute mode of :math:`\\pi \\propto x^{a-1} \\exp^{-P(x)}`,
    with :math:`a>1` and :math:`P` convex polynomial.

    .. math::

        \\pi'(x) = 0
        \\Longleftrightarrow
            X P'(X) - (a-1) = 0
    """

    roots = (np.poly1d([1, 0]) * P.deriv(m=1) - (shape - 1)).roots
    real_roots = roots[np.isreal(roots)].real

    return real_roots[real_roots > 0][0]


def find_a_b_convex_gen_gamma(shape, P, mode):
    """Find solutions :math:`b<0, a>0` of :math:`\\pi(m+x)=\\frac{\\pi(m)}{4}`.

    For :math:`\\pi \\propto x^{\\alpha-1} \\exp^{-(\\beta x^2 + \\gamma x)}`,
     the equation is equivalent to

    .. math::

        (1-\\alpha) \\ln\\left(1+\\frac{x}{m}\\right)
            + \\beta x^2
            + (2\\beta m + \\gamma) x
            - 2\\ln(2)
        = 0
    """

    pol = shift_pol(P, -mode) - P(mode) - 2.0 * np.log(2.0)

    def f(x):
        return (shape - 1.0) * np.log(1.0 + x / mode) - pol(x)

    # Find b < 0 by bisection
    x0_b_left = - (1 - 1e-13) * mode
    x0_b_right = 0.0

    if np.sign(f(x0_b_left)) != np.sign(f(x0_b_right)):
        b = brentq(f, a=x0_b_left, b=x0_b_right)
    else:
        b = -0.5 * mode

    # Find a > 0 using Halley's method
    # initial point
    roots = pol.roots
    x0_a = np.max(roots[np.isreal(roots)].real)

    # 1st and 2nd derivatives
    d1_pol, d2_pol = pol.deriv(m=1), pol.deriv(m=2)

    def d1_f(x):
        return (shape - 1.0) / (mode + x) - d1_pol(x)

    def d2_f(x):
        return -(shape - 1.0) / (mode + x)**2 - d2_pol(x)

    a = newton(f, x0_a, fprime=d1_f, fprime2=d2_f)

    return np.array([b, a])  # b < 0 < a


def trunc_gaussian_sampler(b, mu, var, random_state=None):

    rng = check_random_state(random_state)

    it_max = int(1e7)

    for it in range(1, it_max + 1):
        X = rng.normal(loc=b, scale=np.sqrt(var))
        U = rng.rand()
        if (np.log(U) < -(b - mu) * (X - b) / var) and (X > b):
            break
    else:  # if no acceptation
        print('gen_gamma with alpha=1 not accepted\
                after {} iterations'.format(it))
        X = 1e-7

    return X, it


def gen_gamma_alpha_lt_1_sampler(shape, P, random_state=None):

    rng = check_random_state(random_state)

    it_max = int(1e2)
    for it in range(1, it_max + 1):

        X = rng.gamma(shape=shape, scale=1 / P[1], size=1)
        U = rng.rand()
        if np.log(U) < -P[2] * X**2:
            break

    else:
        print('gen_gamma with alpha<1 not accepted\
                after {} iterations'.format(it))
        pass
        # X = ...

    return X, it


def sampler_exact_convex_quartic(P, shape=None, random_state=None):
    """Sample from

    - shape is None

        .. math::

            \\pi(x)\\propto \\exp^{-(\\alpha x^4 + \\beta x^2 + \\gamma x)}

    - shape not None

        .. math::

            \\pi(x)\\propto x^{\\alpha-1} \\exp^{-(\\beta x**2 + \\gamma x)}

    .. seealso::

        :cite:`Dev12` Equation 2
        "A note on generating random variables with log-concave densities"
        `https://pdfs.semanticscholar.org/e9d3/f6e0862bc6893cdcce47fa6e8380d0fd765e.pdf <https://pdfs.semanticscholar.org/e9d3/f6e0862bc6893cdcce47fa6e8380d0fd765e.pdf>`_
    """

    # Compute mode of target density
    # Define f(x) = log_pi(x + mode) i.e. place the mode at the origin
    # Compute a>0, b<0 satisfying pi(m+x) >= pi(m)/4 >= pi(m+2x)

    rng = check_random_state(random_state)

    if shape is None:  # exp(-(alpha x^4 + beta x^2 + gamma x))
        mode = find_mode_convex_quartic(P)

        def log_f(x):
            return -P(x + mode)

        b, a = 0.5 * find_b_a_convex_quartic(P, mode)

    else:  # x^(shape-1) exp(-(beta x^2 + gamma x))

        if not P[2]:  # x^(shape-1) exp(- gamma x))
            return rng.gamma(shape, 1 / P[1]), 0

        else:
            if shape < 1.0:  # x^(shape-1) exp(-(beta x^2 + gamma x)): no mode
                return gen_gamma_alpha_lt_1_sampler(shape, P,
                                                    random_state=rng)

            elif shape == 1.0 and P[2]:
                # x^0 exp(-(beta x^2 + gamma x)) = truncated Gaussian [0, oo]
                return trunc_gaussian_sampler(b=0.0,
                                              mu=-0.5 * P[1] / P[2],
                                              var=0.5 / P[2],
                                              random_state=rng)

        # else there exists a mode, use [Dev12] domination
        mode = find_mode_convex_gen_gamma(shape, P)

        def log_f(x):
            return (shape - 1) * np.log(x + mode) - P(x + mode)

        b, a = 0.5 * find_a_b_convex_gen_gamma(shape, P, mode)

    _2b, _b, _a, _2a = 2 * b, b, a, 2 * a
    abs_b = abs(b)  # -b since b<0

    log_f_2b, log_f_b, log_f_0, log_f_a, log_f_2a =\
        log_f(np.array([_2b, _b, 0.0, _a, _2a]))
    log_f_b_2b, log_f_a_2a = log_f_b - log_f_2b, log_f_a - log_f_2a

    # Dominating function h(x) >= f(x)
    def log_h(x):
        if x < _2b:  # (2b-x)/b log_f(b) + (x-b)/b log_f(2b)
            return (_b - x) * (log_f_b_2b / _b) + log_f_b
        elif x < _b:  # 2b <= x < b
            return log_f_b
        elif x < _a:  # b <= x < a
            return log_f_0
        elif x < _2a:  # a <= x < 2a
            return log_f_a
        else:  # if x >= _2a, (2a-x)/a log_f(a) + (x-a)/a log_f(2a)
            return (_a - x) * (log_f_a_2a / _a) + log_f_a

    # Compute normalizing constant Z_h = int h
    # Z_h = int_[b, a] f(0)
    #     + int_[a,2a] f(a) + int_[2b,b] f(b)
    #     + int_[2a, +oo] exp.. + int_[-oo, 2b] exp..
    #
    # Z_h -= int_[-m, 2b] exp.. for gen_gamma

    a_lse = [log_f_0, log_f_a, log_f_b, log_f_2a, log_f_2b]
    b_lse = [_a + abs_b, _a, abs_b, _a / log_f_a_2a, abs_b / log_f_b_2b]

    if shape is not None:
        a_lse.append(log_f_b + (1 + mode / _b) * log_f_b_2b)
        b_lse.append(-abs_b / log_f_b_2b)
        trunc = mode + _2b
    else:
        trunc = np.inf

    log_Z_h = logsumexp(a=a_lse, b=b_lse)

    nb_iter_max = 100
    for iter_ in range(nb_iter_max):

        U, V, W = rng.rand(3)  # Unif_[0,1]
        log_V = np.log(V) + log_Z_h  # Incorporate Z_h in the LHS

        # Proposal mechanism:
        # 1. Choose a region C with proba 1/Z_h * \\int_C h (multinomial)
        # 2. Sample X ~ h restricted to C (uniform | exponential tails)

        if log_V <= logsumexp(a=a_lse[:1], b=b_lse[:1]):
            # propto int_[b, a] h = (a-b) f(0) = (a+|b|) f(0)
            # X ~ U([b, a])
            X = _b + (_a - _b) * U
        elif log_V <= logsumexp(a=a_lse[:2], b=b_lse[:2]):
            # propto int_[a, 2a] h = a f(a)
            # X ~ U([a, 2a])
            X = _a + _a * U
        elif log_V <= logsumexp(a=a_lse[:3], b=b_lse[:3]):
            # propto int_[2b, b] h = -b f(b) = |b| f(b)
            # X ~ U([2b, b])
            X = _b + _b * U
        elif log_V <= logsumexp(a=a_lse[:4], b=b_lse[:4]):
            # propto int_[2a, +oo] h = ...
            # X ~ 2a + Exp(log(f(a)/f(2a))/a)
            mu = _a / log_f_a_2a
            X = _2a - mu * np.log(U)
        else:
            # propto int_[-(oo or mode), 2b] h
            # X ~ 2b - Exp(log(f(b)/f(2b))/|b|)
            mu = abs_b / log_f_b_2b
            X = _2b + mu * np.log(1.0 - U * (1.0 - np.exp(-trunc / mu)))

        # Accept/reject X ~ h/Z_h with proba f(X)/h(X) = pi(X+m)/h(X)
        if np.log(W) <= log_f(X) - log_h(X):
            return X + mode, iter_

    else:
        print('Dev12 not accepted after {} iterations!'.format(iter_))
        return X + mode, iter_


def sampler_mala(x, V, sigma=0.01, nb_steps=20, random_state=None):
    """ Sample from :math:`\\pi(x) \\propto \\exp(-V(x))` with :math:`V` polynomial using Metropolis Adjusted Langevin Algorithm (MALA)
    """

    rng = check_random_state(random_state)

    x_ = x
    d1_V = V.deriv(m=1)

    for _ in range(nb_steps):

        Vx, grad_Vx = V(x_), d1_V(x_)
        y = x_ - 0.5 * sigma**2 * grad_Vx + sigma * rng.randn()
        Vy, grad_Vy = V(y), d1_V(y)

        acceptance = Vx - Vy\
                    + 0.5\
                        / sigma**2\
                        * ((x_ - y + 0.5 * sigma**2 * grad_Vy)**2
                            - (y - x_ + 0.5 * sigma**2 * grad_Vx)**2)

        if np.log(rng.rand()) < acceptance:
            x_ = y

    return x_


def polynomial_in_negative_log_conditional_a_coef(i, a, b, V):

    if len(V) > 7:
        str_ = ['Polynomial potentials V are allowed up to degree 6',
                'Given\n',
                ' '.join(['g_{}={}'.format(n, g_n)
                          for n, g_n in enumerate(V)])]
        raise ValueError(' '.join(str_))

    P = np.poly1d(0.0)
    i_2 = max(0, i - 2)

    if V[1]:
        P += V[1]\
            * np.poly1d(
                [1.0,  # X
                 0.0])  # 1

    if V[2]:
        P += V[2]\
            * np.poly1d(
                [1.0,  # X^2
                 0.0,  # X
                 0.0])  # 1

    if V[3]:
        P += V[3]\
            * np.poly1d(
                [1.0,  # X^3
                 0.0,  # X^2
                 3 * (b[i-1] + b[i]),  # X
                 0.0])  # 1

    if V[4]:
        P += V[4]\
            * np.poly1d(
                [1.0,  # X^4
                 0.0,  # X^3
                 4 * (b[i-1] + b[i]),  # X^2
                 4 * (a[i-1] * b[i-1] + a[i+1] * b[i]),  # X
                 0.0])  # 1

    if V[5]:
        P += V[5]\
            * np.poly1d(
                [1.0,  # X^5
                 0.0,  # X^4
                 5 * (b[i-1] + b[i]),  # X^3
                 5 * (a[i-1] * b[i-1] + a[i+1] * b[i]),  # X^2
                 5 * (b[i-1] * (a[i-1]**2 + b[i_2])
                        + (b[i-1] + b[i])**2
                        + b[i] * (a[i+1]**2 + b[i+1])),  # X
                 0.0])  # 1

    if V[6]:
        P += V[6]\
            * np.poly1d(
                [1.0,  # X^6
                 0.0,  # X^5
                 6 * (b[i-1] + b[i]),  # X^4
                 6 * (a[i-1] * b[i-1] + a[i+1] * b[i]),  # X^3
                 3 * (2 * b[i-1] * (a[i-1]**2 + b[i_2])
                        + 3 * (b[i-1] + b[i])**2
                        + 2 * b[i] * (a[i+1]**2 + b[i+1])),  # X^2
                 6 * (a[i_2] * b[i_2] * b[i-1]
                        + a[i-1] * b[i-1]
                                 * (a[i-1]**2 + 2*(b[i_2] + b[i-1] + b[i]))
                        + a[i+1] * b[i]
                                 * (a[i+1]**2 + 2*(b[i-1] + b[i] + b[i+1]))
                        + a[i+2] * b[i] * b[i+1]),  # X
                 0.0])  # 1

    return P


def polynomial_in_negative_log_conditional_b_coef(i, a, b, V):

    if len(V) > 7:
        str_ = ['Polynomial potentials V are allowed up to degree 6',
                'Given\n',
                ' '.join(['g_{}={}'.format(n, g_n)
                          for n, g_n in enumerate(V)])]
        raise ValueError(' '.join(str_))

    P = np.poly1d(0.0)

    if V[2]:
        P += V[2]\
            * np.poly1d(
                [2.0,  # X
                 0.0])  # 1

    if V[3]:
        P += V[3]\
            * np.poly1d(
                [3 * (a[i] + a[i+1]),  # X
                 0.0])  # 1

    if V[4]:
        P += V[4]\
            * np.poly1d(
                [2.0,  # X^2
                 4 * (a[i]**2 + a[i] * a[i+1] + a[i+1]**2
                        + b[i-1] + b[i+1]),  # X
                 0.0])  # 1

    if V[5]:
        P += V[5]\
            * np.poly1d(
                [5 * (a[i] + a[i+1]),  # X^2
                 5 * (a[i]**3 + a[i]**2 * a[i+1] + a[i] * a[i+1]**2 + a[i+1]**3
                        + b[i-1] * (a[i-1] + 2 * a[i] + a[i+1])
                        + b[i+1] * (a[i] + 2 * a[i+1] + a[i+2])),  # X
                 0.0])  # 1

    if V[6]:
        P += V[6]\
            * np.poly1d(
                [2.0,  # X^3
                 3 * (3 * a[i]**2 + 4 * a[i] * a[i+1] + 3 * a[i+1]**2
                        + 2 * (b[i-1] + b[i+1])),  # X^2
                 6 * (a[i]**4 + a[i]**3 * a[i+1] + a[i]**2 * a[i+1]**2
                        + a[i] * a[i+1]**3 + a[i+1]**4
                        + b[i-1] * (a[i-1]**2 + a[i-1] * a[i+1] + a[i+1]**2
                                    + a[i] * (2*a[i-1]+ 3*a[i] + 2*a[i+1])
                                    + b[i-2] + b[i-1] + b[i+1])
                        + b[i+1] * (a[i]**2 + a[i]*a[i+2] + a[i+2]**2
                                    + a[i+1] * (2*a[i]+ 3*a[i+1] + 2*a[i+2])
                                    + b[i+1] + b[i+2])),  # X
                 0.0])  # 1

    return P

"""Equilibrium densities"""


def equilibrium_x2_x4(g_2, g_4):
    """ The equilibrium measure associated to the :math:`\\beta`-ensemble with potential

    .. math::

        V(x) = V_t(x) = \\frac{g_2}{2} x^2 + \\frac{g_4}{4} x^4

    takes the form

    .. math::

        d \\mu(x) = \\frac{g_4}{2\\pi}(x^2 + b^2)\\sqrt{(a^2 - x^2)_+} \\ dx

    where :math:`a^2 = \\frac{2}{3g_4}\\left(\\sqrt{g_2^2+12g_4}-g_2\\right)` and :math:`b^2 = \\frac{1}{3g_4}\\left(\\sqrt{g_2^2+12g_4}+2g_2\\right)`

    .. seealso::

        - :cite:`DuKu06` p.2-3 `https://arxiv.org/pdf/math/0605201.pdf <https://arxiv.org/pdf/math/0605201.pdf>`_
        - :cite:`Mol` Example 3.3 `http://pcteserver.mi.infn.it/~molinari/RMT/RM2.pdf <http://pcteserver.mi.infn.it/~molinari/RMT/RM2.pdf>`_
    """

    if g_2 > -4 * np.sqrt(g_4 / 4):

        a2 = 2 / (3 * g_4) * (np.sqrt(g_2**2 + 12 * g_4) - g_2)
        b2 = 1 / (3 * g_4) * (np.sqrt(g_2**2 + 12 * g_4) + 2 * g_2)

        def dmu(x):
            return np.maximum(g_4 / (2 * np.pi)
                                * (x**2 + b2)
                                * np.sqrt(np.maximum(a2 - x**2, 0)),
                              0)

    else:
        b2, a2 = (-g_2 + 4 * np.sqrt(g_4 / 4) * np.array([-1, 1])) / g_4

        def dmu(x):
            return g_4 / (2 * np.pi)\
                       * np.abs(x)\
                       * np.sqrt(np.maximum((a2 - x**2)*(x**2 - b2), 0))

    support = np.sqrt(a2) * np.array([-1, 1])

    return dmu, support


def cdf_equilibrium_x2_x4(g_2, g_4):
    """ Cumulative distribution function of the equilibrium distribution associated to the :math:`\\beta`-ensemble with potential

    .. math::

        V(x) = V_t(x) = \\frac{g_2}{2} x^2 + \\frac{g_4}{4} x^4
    """

    a2 = 2 / (3 * g_4) * (np.sqrt(g_2**2 + 12 * g_4) - g_2)
    a = np.sqrt(a2)

    b2 = 1 / (3 * g_4) * (np.sqrt(g_2**2 + 12 * g_4) + 2 * g_2)

    def cdf(x):

        x[x < -a] = -a
        x[x > a] = a
        X = np.arcsin(x / a)
        # return g_4 * a2 / (8 * np.pi) * ((X + np.pi) * (a2 + 4 * b2) / 4
        #                                  - a2 * np.sin(4 * X) / 8
        #                                  + b2 * np.sin(2 * X))

        return g_4 * a2 / (8 * np.pi)\
                * ((a2 + 4 * b2) / 2 * (X + np.pi / 2)
                   + b2 * np.sin(2 * X)
                   - a2 / 8 * np.sin(4 * X))

    support = a * np.array([-1, 1])

    return cdf, support


def equilibrium_x2m(m, g_2m):
    """ The equilibrium measure associated to the :math:`\\beta`-ensemble with potential

    .. math::

        V(x) = \\underbrace{\\frac{g_{2m}}{2m}}_{=t} x^{2m}

    takes the form

    .. math::

        d \\mu(x) = \\frac{mt}{\\pi} \\sqrt{(a^2 - x^2)_+} h_1(x) \\ dx

    where
    :math:`a^2 = \\left( m t \\prod_{l=1}^{m} \\frac{2l-1}{2l} \\right)^{-\\frac{1}{m}}`
    and
    :math:`h_1(x) = x^{2(m-1)} + \\sum_{j=1}^{m-1} x^{2(m-j-1)} a^{2j} \\prod_{l=1}^{j} \\frac{2l - 1}{2l}`

    .. seealso::

        - :cite:`Dei00` Proposition 6.156 "Orthogonal polynomials and random matrices : a Riemann-Hilbert approach", ISBN 0821826956.
    """

    t = g_2m / (2 * m)
    c_prod = np.cumprod([(2 * l - 1) / (2 * l) for l in range(1, m + 1)])
    a2 = 1 / (m * t * c_prod[-1])**(1 / m)

    h1 = np.poly1d(0.0)
    h1[2 * (m - 1)] = 1.0

    if m > 1:
        h1._coeffs[2::2] = a2**np.arange(1, m) * c_prod[:-1]

    def mu_eq(x):
        return np.maximum(m * t / np.pi
                            * np.sqrt(np.maximum(a2 - x**2, 0))
                            * h1(x),
                          0)

    support = np.sqrt(a2) * np.array([-1, 1])

    return mu_eq, support


def cdf_equilibrium_x2m(m, g_2m):
    """ Cumulative distribution function of the equilibrium distribution associated to the :math:`\\beta`-ensemble with potential

    .. math::

        V(x) = \\underbrace{\\frac{g_{2m}}{2m}}_{=t} x^{2m}

    """

    t = g_2m / (2 * m)
    c_prod = np.cumprod([(2 * l - 1) / (2 * l) for l in range(1, m + 1)])
    a2 = 1 / (m * t * c_prod[-1])**(1 / m)

    a = np.sqrt(a2)

    if m == 2:
        def cdf(x):

            x[x < -a] = -a
            x[x > a] = a
            X = np.arcsin(x / a)
            return m * t * a**4 / (8 * np.pi) \
                   * (3 * (X + np.pi / 2)
                      + np.sin(2 * X)
                      - 1 / 4 * np.sin(4 * X))

    elif m == 3:
        def cdf(x):

            x[x < -a] = -a
            x[x > a] = a
            X = np.arcsin(x / a)
            return m * t * a**6 / (16 * np.pi) \
                   * (5 * (X + np.pi / 2)
                      + 5 / 4 * np.sin(2 * X)
                      - 1 / 2 * np.sin(4 * X)
                      + 1 / 12 * np.sin(6 * X))

    else:
        raise NotImplementedError('m != 2, 3')

    support = a * np.array([-1, 1])

    return cdf, support


def equilibrium_ClItKr10():
    """.. math::

        V(x) = \\frac{1}{20} x^4 - \\frac{4}{15}x^3 + \\frac{1}{5}x^2 + \\frac{8}{5}x

    .. math::

        d \\mu(x) = \\frac{1}{10\\pi} \\sqrt{[(x+2)(2-x)^5]_+} \\ dx

    .. seealso::

        - :cite:`ClItKr10` Example 1.2 HIGHER-ORDER ANALOGUES OF THE TRACY-WIDOM DISTRIBUTION
        - :cite:`OlNaTr14` Section 3.2 https://arxiv.org/pdf/1404.0071.pdf
    """

    def mu_eq(x):
        return 1 / (10 * np.pi) * np.sqrt(np.maximum((x + 2) * (2 - x)**5, 0))

    support = 2.0 * np.array([-1, 1])

    return mu_eq, support


class TracyWidom(object):
    """ Implements computation of the cumulative distribution function (CDF) of the Tracy Widom 2 distribution based on the work of Bornemann [2010] https://arxiv.org/pdf/0804.2543.pdf

    .. seealso::

        :ref:`another implementation <https://gist.github.com/yymao/7282002>`
    """
    def __init__(self):
        self.quad_order = None
        self.x_quad, self.w_quad = None, None

    def change_of_variable(self, s):  # , old_implem=False):
        """Bornemann Equation 7.5"""

        def phi(x):
            # if old_implem:
            #     return s + 10 * np.tan(0.5 * np.pi * x)
            # else:
            return np.atleast_1d(s)[:, None] + 10 * np.tan(0.5 * np.pi * x)

        def d_phi(x):
            return 10 * np.pi / 2 * (1 + np.tan(0.5 * np.pi * x)**2)

        return phi, d_phi

    def kernel(self, s):  # , old_implem=False):
        """ operator is restricted to L^2(0,1), so you need to change variables first
        """
        phi, d_phi = self.change_of_variable(s)

        # if old_implem:

        #     def K(x, y):
        #         K_Ai_xy = self.airy_kernel(phi(x), phi(y))

        #         return np.sqrt(d_phi(x) * d_phi(y)) * K_Ai_xy

        # else:

        def K(X, Y=None, eps_xy=1e-5):
            # Compute K[x, y]_{x, y \in X}
            # K_xy = (A_x * dA_y - dA_x * A_y) / (x - y)
            X = np.ravel(X)
            size_X = len(X)

            x = phi(X)
            sqrt_dphi_x = np.sqrt(d_phi(X))
            A_x, dA_x, _, _ = airy(x)
            # y = x
            A_y, dA_y = A_x, dA_x

            outer = 'ki, kj -> kij'
            K_xy = np.einsum(outer, A_x, dA_y) - np.einsum(outer, dA_x, A_y)

            # divide upper triangular part and symmetrize
            # K_xy /= x_y, where x_y = [x_i - y_j]_ij (recall y = x)
            x_y = x[0][:, None] - x[0]
            idx = np.triu_indices_from(x_y, k=1)

            K_xy[:, idx[0], idx[1]] /= x_y[idx[0], idx[1]]  # K_xy /= x-y
            K_xy[:, idx[1], idx[0]] = K_xy[:, idx[0], idx[1]]  # K_yx = K_xy

            # fill the diagonal K_xx
            # set y = x - eps
            # K_xx = (A_x * dA_x-eps - dA_x * A_x-eps) / eps
            A_y, dA_y, _, _ = airy(x - eps_xy)
            diag = range(size_X)
            K_xy[:, diag, diag] = (A_x * dA_y - dA_x * A_y) / eps_xy

            return sqrt_dphi_x[:, None] * K_xy * sqrt_dphi_x

        return K

    def cdf(self, x, quad_order=50):  # , old_implem=False):
        """Compute cumulative distribution function :math:`F_{TW_2}` of the Tracy-Widom :math:`TW_2` distribution

        :param x: points where
        :type x: float or 1D array-like

        :param quad_order: order (degree) of the quadrature used to compute the underlying Fredholm determinant.
        :type quad_order: int
        """

        if self.quad_order != quad_order:
            self.quad_order = quad_order
            self.x_quad, self.w_quad = self.compute_quadrature(quad_order)

        return self.fredholm_determinant(self.kernel(x),
                                         self.x_quad,
                                         self.w_quad)  # , old_implem)

    @staticmethod
    def airy_kernel(x, y, eps_xy=1e-5):
        """evaluate Airy kernel pointwise
        https://en.wikipedia.org/wiki/Tracy%E2%80%93Widom_distribution
        """

        # hack to extend the kernel by continuity on the diagonal
        if x == y:
            y += eps_xy

        Ai_x, dAi_x, _, _ = airy(x)
        Ai_y, dAi_y, _, _ = airy(y)

        return (Ai_x * dAi_y - Ai_y * dAi_x) / (x - y)

    @staticmethod
    def compute_quadrature(order, type='Legendre'):

        x, w = np.polynomial.legendre.leggauss(order)
        return 0.5 * (x + 1), 0.5 * w

    @staticmethod
    def fredholm_determinant(kernel, x_quad, w_quad):  # , old_implem):
        """ This implements the numerical evaluation of Fredholm determinants as in [Bornemann, 2010] Equation 6.1

        .. math::

            \\det (I - K)_L^2[0,1] \\approx \\det (I - [K(x, y)]_xy)
        """

        # if old_implem:
        #     K = np.array([[kernel(x, y) for x in x_quad] for y in x_quad])
        # else:
        K = kernel(x_quad, x_quad)
        shp = K.shape

        d = np.sqrt(w_quad)

        Id = np.broadcast_to(np.eye(*shp[1:]), shp)

        return np.linalg.det(Id - d[:, None] * K * d)
