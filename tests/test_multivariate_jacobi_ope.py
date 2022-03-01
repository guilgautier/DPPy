import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import eval_jacobi

from dppy.continuous.jacobi import norm_jacobi
from dppy.continuous.utils import compute_ordering


@pytest.mark.parametrize(
    "expected",
    [
        [
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
            [0, 2],
            [1, 2],
            [2, 0],
            [2, 1],
            [2, 2],
            [0, 3],
            [1, 3],
            [2, 3],
            [3, 0],
            [3, 1],
            [3, 2],
            [3, 3],
        ],
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
            [0, 1, 1],
            [1, 0, 0],
            [1, 0, 1],
            [1, 1, 0],
            [1, 1, 1],
            [0, 0, 2],
            [0, 1, 2],
            [0, 2, 0],
            [0, 2, 1],
            [0, 2, 2],
            [1, 0, 2],
            [1, 1, 2],
            [1, 2, 0],
            [1, 2, 1],
            [1, 2, 2],
            [2, 0, 0],
            [2, 0, 1],
            [2, 0, 2],
            [2, 1, 0],
            [2, 1, 1],
            [2, 1, 2],
            [2, 2, 0],
            [2, 2, 1],
            [2, 2, 2],
        ],
    ],
)
def test_ordering_BardenetHardy(expected):
    N, d = np.shape(expected)
    computed = compute_ordering(N, d)
    np.testing.assert_array_equal(computed, expected)


@pytest.mark.parametrize(
    "a, b",
    [
        (0.5, 0.5),
        (1, 1),
        (-0.3, 0.4),
    ],
)
def test_norm_of_multiD_jacobi_polynomials(a, b):

    w = lambda x, a, b: (1 - x) ** a * (1 + x) ** b
    P = lambda n, a, b, x: eval_jacobi(n, a, b, x)

    def norm_quad(k, a, b):
        norm2 = quad(lambda x: w(x, a, b) * P(k, a, b, x) ** 2, -1, 1)[0]
        return np.sqrt(norm2)

    orders = np.arange(100)

    computed = [norm_jacobi(n, a, b) for n in orders]
    expected = [norm_quad(n, a, b) for n in orders]

    np.testing.assert_array_almost_equal(computed, expected)
