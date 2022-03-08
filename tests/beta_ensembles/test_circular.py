import numpy as np
import pytest

from dppy.beta_ensembles.cicular import (
    sampler_circular_full,
    sampler_circular_quindiagonal,
)


@pytest.mark.parametrize("beta", [1, 2])
def test_all_points_lie_on_unit_circle_full_matrix_model(beta):
    sample = sampler_circular_full(beta, 1000)
    np.testing.assert_array_almost_equal(np.abs(sample), 1.0)


@pytest.mark.parametrize("beta", [0, 1, 2, 4, 5, 10, 30])
def test_all_points_lie_on_unit_circle_quindiagonal_matrix_model(beta):
    sample = sampler_circular_quindiagonal(beta, 1000)
    np.testing.assert_array_almost_equal(np.abs(sample), 1.0)
