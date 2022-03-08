import numpy as np
import scipy.linalg as la
from scipy.stats import semicircular


# hermite
def semi_circle_pdf(x, loc=0.0, scale=2.0):
    return semicircular.pdf(x, loc=loc, scale=scale)


def semi_circle_support(loc=0.0, scale=2.0):
    return loc - scale, loc + scale


# Laguerre
def marcencko_pastur_pdf(x, c, sigma=1.0):
    a, b = marcencko_pastur_support(c, sigma=sigma)
    pdf_x = np.sqrt(np.maximum((b - x) * (x - a), 0))
    pdf_x /= c * x
    pdf_x /= 2 * np.pi * sigma ** 2
    return pdf_x


def marcencko_pastur_support(c, sigma=1.0):
    a = (sigma * (1 - np.sqrt(c))) ** 2
    b = (sigma * (1 + np.sqrt(c))) ** 2
    return (a, b)


# Jacobi
def wachter_pdf(x, a, b):
    """
    .. seealso::

        :cite:`DuEd15` Table 1
    """
    Lm, Lp = wachter_support(a, b)
    pdf_x = np.sqrt(np.maximum((Lp - x) * (x - Lm), 0.0))
    pdf_x /= x * (1 - x)
    pdf_x *= (a + b) / (2 * np.pi)

    return pdf_x


def wachter_support(a, b):
    assert a >= 1 and b >= 1
    Lm = ((np.sqrt(a * (a + b - 1)) - np.sqrt(b)) / (a + b)) ** 2
    Lp = ((np.sqrt(a * (a + b - 1)) + np.sqrt(b)) / (a + b)) ** 2
    return (Lm, Lp)


# Circular
def uniform_unit_circle_pdf(self, x):
    return np.where(la.norm(x, axis=-1) == 1, 0.5 / np.pi, 0.0)


# Ginibre
def uniform_unit_disk_pdf(x):
    return np.where(la.norm(x, axis=-1) <= 1, 1.0 / np.pi, 0.0)


# Polynomial potentials

# V(x) = g4 / 4 x^4 + g2 / 2 x^2
def equilibrium_x2_x4_pdf(x, g_2, g_4):
    r"""The equilibrium measure associated to the :math:`\beta`-ensemble with potential

    .. math::

        V(x) = V_t(x) = \frac{g_2}{2} x^2 + \frac{g_4}{4} x^4

    takes the form

    .. math::

        d \mu(x) = \frac{g_4}{2\pi}(x^2 + b^2)\sqrt{(a^2 - x^2)_+} \ dx

    where :math:`a^2 = \frac{2}{3g_4}\left(\sqrt{g_2^2+12g_4}-g_2\right)` and :math:`b^2 = \frac{1}{3g_4}\left(\sqrt{g_2^2+12g_4}+2g_2\right)`

    .. seealso::

        - :cite:`DuKu06` p.2-3 `https://arxiv.org/pdf/math/0605201.pdf <https://arxiv.org/pdf/math/0605201.pdf>`_
        - :cite:`Mol` Example 3.3 `http://pcteserver.mi.infn.it/~molinari/RMT/RM2.pdf <http://pcteserver.mi.infn.it/~molinari/RMT/RM2.pdf>`_
    """

    if g_2 > -2 * np.sqrt(g_4):
        a2 = 2 / (3 * g_4) * (np.sqrt(g_2 ** 2 + 12 * g_4) - g_2)
        b2 = 0.5 * a2 + g_2 / g_4

        pdf_x = np.sqrt(np.maximum((a2 - x ** 2), 0))
        pdf_x *= x ** 2 + b2
        pdf_x *= g_4 / (2 * np.pi)

    else:
        a2 = (-g_2 + 2 * np.sqrt(g_4)) / g_4
        b2 = (-g_2 - 2 * np.sqrt(g_4)) / g_4

        pdf_x = np.sqrt(np.maximum((a2 - x ** 2) * (x ** 2 - b2), 0))
        pdf_x *= np.abs(x)
        pdf_x *= g_4 / (2 * np.pi)

    return pdf_x


def equilibrium_x2_x4_support(g_2, g_4):
    if g_2 > -2 * np.sqrt(g_4):
        a = np.sqrt(2 / (3 * g_4) * (np.sqrt(g_2 ** 2 + 12 * g_4) - g_2))
        return (-a, a)
    else:
        a = np.sqrt((-g_2 + 2 * np.sqrt(g_4)) / g_4)
        b = np.sqrt((-g_2 - 2 * np.sqrt(g_4)) / g_4)
        return ((-a, -b), (b, a))


def equilibrium_x2_x4_cdf(x, g_2, g_4):
    assert g_2 > -2 * np.sqrt(g_4)

    a2 = 2 / (3 * g_4) * (np.sqrt(g_2 ** 2 + 12 * g_4) - g_2)
    b2 = 0.5 * a2 + g_2 / g_4

    a = np.sqrt(a2)
    Y = np.arcsin(np.clip(x, -a, a) / a)

    tmp2 = 4 * Y
    tmp2 -= np.sin(4 * Y)
    tmp2 += 2 * np.pi
    tmp2 *= a2 ** 2 / 32

    tmp0 = 2 * Y
    tmp0 += np.sin(2 * Y)
    tmp0 += np.pi
    tmp0 *= a2 / 4

    cdf_x = g_4 / (2 * np.pi) * (tmp2 + b2 * tmp0)

    return cdf_x


# V(x) = g / (2m) x^2m
def equilibrium_x2m_pdf(x, m, g_2m):
    r"""The equilibrium measure associated to the :math:`\beta`-ensemble with potential

    .. math::

        V(x) = \underbrace{\frac{g_{2m}}{2m}}_{=t} x^{2m}

    takes the form

    .. math::

        d \mu(x) = \frac{mt}{\pi} \sqrt{(a^2 - x^2)_+} h_1(x) \ dx

    where
    :math:`a^2 = \left( m t \prod_{l=1}^{m} \frac{2l-1}{2l} \right)^{-\frac{1}{m}}`
    and
    :math:`h_1(x) = x^{2(m-1)} + \sum_{j=1}^{m-1} x^{2(m-j-1)} a^{2j} \prod_{l=1}^{j} \frac{2l - 1}{2l}`

    .. seealso::

        - :cite:`Dei00` Proposition 6.156 "Orthogonal polynomials and random matrices : a Riemann-Hilbert approach", ISBN 0821826956.
    """
    t = g_2m / (2 * m)
    c_prod = np.cumprod([(2 * l - 1) / (2 * l) for l in range(1, m + 1)])
    a2 = 1 / (m * t * c_prod[-1]) ** (1 / m)

    # h1(x) = x^2(m-1) + sum_j=1^m-1 c_j x^ 2(m-j-1)
    coeffs = np.zeros(2 * (m - 1) + 1)
    coeffs[-1] = 1.0
    for j in range(1, m):
        coeffs[2 * (m - j - 1)] = c_prod[j - 1] * a2 ** j

    h1 = np.polynomial.Polynomial(coeffs)

    pdf_x = np.sqrt(np.maximum(a2 - x ** 2, 0))
    pdf_x *= h1(x)
    pdf_x *= m * t / np.pi

    return pdf_x


def equilibrium_x2m_support(m, g_2m):
    t = g_2m / (2 * m)
    c_prod = np.cumprod([(2 * l - 1) / (2 * l) for l in range(1, m + 1)])
    a = np.sqrt(1 / (m * t * c_prod[-1]) ** (1 / m))
    return (-a, a)


def equilibrium_x2m_cdf(x, m, g_2m):
    assert m in (1, 2, 3, 4)

    t = g_2m / (2 * m)
    c_prod = np.cumprod([(2 * l - 1) / (2 * l) for l in range(1, m + 1)])
    a2 = 1 / (m * t * c_prod[-1]) ** (1 / m)

    # cdf(x) = (m * t / np.pi) int sqrt{(a^2 - x^2)_+} h_1(x)
    cdf_x = np.zeros(len(x), dtype=float)
    # h1(x) = x^2(m-1) + sum_j=1^m-1 c_j x^ 2(m-j-1)
    coeffs = np.ones(m)
    coeffs[1:] = a2 ** np.arange(1, m) * c_prod[:-1]

    # change of variable sin(y) = x / a
    a = np.sqrt(a2)
    y = np.arcsin(np.clip(x, -a, a) / a)

    for i, c_i in enumerate(coeffs, start=1):
        if (m - i) == 3:
            tmp = 120 * y
            tmp -= 48 * np.sin(2 * y)
            tmp -= 24 * np.sin(4 * y)
            tmp += 16 * np.sin(6 * y)
            tmp -= 3 * np.sin(8 * y)
            tmp += 60 * np.pi
            tmp *= a2 ** 4 * c_i / 3072

        if (m - i) == 2:
            tmp = 12 * y
            tmp -= 3 * np.sin(2 * y)
            tmp -= 3 * np.sin(4 * y)
            tmp += np.sin(6 * y)
            tmp += 6 * np.pi
            tmp *= a2 ** 3 * c_i / 192

        if (m - i) == 1:
            tmp = 4 * y
            tmp -= np.sin(4 * y)
            tmp += 2 * np.pi
            tmp *= a2 ** 2 * c_i / 32

        if (m - i) == 0:
            tmp = 2 * y
            tmp += np.sin(2 * y)
            tmp += np.pi
            tmp *= a2 * c_i / 4

        cdf_x += tmp

    cdf_x *= m * t / np.pi
    return cdf_x


# V(x) = 1/20 x^4 - 4/15 x^3 + 1/5 x^2 + 8/5 x
def equilibrium_ClItKr10_pdf(x):
    r""".. math::

        V(x) = \frac{1}{20} x^4 - \frac{4}{15}x^3 + \frac{1}{5}x^2 + \frac{8}{5}x

    .. math::

        d \mu(x) = \frac{1}{10\pi} \sqrt{[(x+2)(2-x)^5]_+} \ dx

    .. seealso::

        - :cite:`ClItKr10` Example 1.2 HIGHER-ORDER ANALOGUES OF THE TRACY-WIDOM DISTRIBUTION
        - :cite:`OlNaTr14` Section 3.2 https://arxiv.org/pdf/1404.0071.pdf
    """
    pdf_x = np.sqrt(np.maximum((x + 2) * (2 - x) ** 5, 0))
    pdf_x /= 10 * np.pi
    return pdf_x


def equilibrium_ClItKr10_support():
    return (-2.0, 2.0)
