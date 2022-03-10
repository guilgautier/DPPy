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


def norm_jacobi_quad(n, a, b):
    w = lambda x: (1 - x) ** a * (1 + x) ** b
    P = lambda x: eval_jacobi(n, a, b, x)
    norm2 = quad(lambda x: w(x) * P(x) ** 2, -1, 1)[0]
    return np.sqrt(norm2)


@pytest.mark.parametrize(
    "n",
    range(50),
)
@pytest.mark.parametrize(
    "a, b",
    [
        (0.5, 0.5),
        (1, 1),
        (-0.3, 0.4),
    ],
)
def test_norm_of_jacobi_polynomials(a, b, n):
    computed = norm_jacobi(n, a, b)
    expected = norm_jacobi_quad(n, a, b)
    np.testing.assert_almost_equal(computed, expected, decimal=5)
