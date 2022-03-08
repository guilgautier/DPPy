import numpy as np
import pytest

from dppy.beta_ensembles.jacobi import sampler_jacobi_full, sampler_jacobi_tridiagonal


@pytest.mark.parametrize("beta", [1, 2, 4])
@pytest.mark.parametrize(
    "n, m1, m2",
    [
        [100, 100, 100],
        [100, 150, 120],
        [100, 120, 150],
    ],
)
def test_all_points_fall_in_between_0_and_1_full_matrix_model(beta, n, m1, m2):
    sample = sampler_jacobi_full(beta, n, m1, m2)
    np.testing.assert_array_equal(0.0 <= sample, sample <= 1.0)


@pytest.mark.parametrize("beta", [0, 1, 2, 4, 5, 10, 30])
@pytest.mark.parametrize("size", [10, 50, 100, 300])
@pytest.mark.parametrize(
    "a, b",
    [
        [1.0, 1.0],  # uniform
        [0.5, 0.5],  # arcsine
    ],
)
def test_all_points_fall_in_between_0_and_1_tridiagonal_matrix_model(beta, size, a, b):
    sample = sampler_jacobi_tridiagonal(beta, size, a, b)
    np.testing.assert_array_equal(0.0 <= sample, sample <= 1.0)
