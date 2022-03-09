import numpy as np
import numpy.linalg as la
import pytest
from scipy.linalg import qr

from dppy.utils import (
    check_equal_to_O_or_1,
    check_full_row_rank,
    check_geq_0,
    check_hermitian,
    check_in_01,
    check_orthonormal_columns,
    check_projection,
)


def run_test_check(function, expected, array):
    if expected:
        function(array)
        assert True
    else:
        with pytest.raises(ValueError):
            function(array)


def random_array_gaussian(*shape):
    return np.random.randn(*shape)


def random_array_uniform(*shape):
    return np.random.rand(*shape)


#######

X = random_array_gaussian(20, 20)
Y = random_array_gaussian(20, 20)
Z = X + 1j * Y


@pytest.mark.parametrize(
    "expected, array",
    [
        (True, None),
        (False, X),
        (False, Z),
        (True, X.T + X),
        (True, Z.conj().T + Z),
        (True, X.T.dot(X)),
        (False, X.T.dot(Y)),
        (True, Z.conj().T.dot(Z)),
        (False, Z.conj().T.dot(Y)),
    ],
)
def test_check_hermitian(expected, array):
    run_test_check(check_hermitian, expected, array)


#######

X = random_array_gaussian(30, 10)
U, _ = qr(X, mode="economic")


@pytest.mark.parametrize(
    "expected, array",
    [
        (True, None),
        (False, X.dot(X.T)),
        (True, X.dot(la.inv(X.T.dot(X)).dot(X.T))),
        (True, U.dot(U.T)),
    ],
)
def test_projection(expected, array):
    run_test_check(check_projection, expected, array)


#######

X = random_array_gaussian(30, 10)
U, _ = qr(X, mode="economic")


@pytest.mark.parametrize(
    "expected, array",
    [
        (True, None),
        (False, X),
        (True, U),
    ],
)
def test_check_orthonormal_columns(expected, array):
    run_test_check(check_orthonormal_columns, expected, array)


#######


@pytest.mark.parametrize(
    "expected, array",
    [
        (True, None),
        (True, random_array_gaussian(10, 30)),
        (False, random_array_gaussian(31, 30)),
        (False, np.zeros((2, 10))),
        (False, np.ones((2, 20))),
    ],
)
def test_check_full_row_rank(expected, array):
    run_test_check(check_full_row_rank, expected, array)


#######

N = 100
tol = 1e-8


@pytest.mark.parametrize(
    "expected, array",
    [
        (True, None),
        (True, np.ones(N)),
        (True, np.zeros(N)),
        (True, -tol * np.ones(N)),
        (True, tol + np.ones(N)),
        (False, -2 * tol * np.ones(N)),
        (False, np.ones(N) + 2 * tol),
        (False, random_array_uniform(N)),
        (False, random_array_uniform(N, N)),
        (False, 1 - random_array_uniform(N)),
        (False, 1 + random_array_uniform(N)),
        (False, -tol + random_array_uniform(N)),
    ],
)
def test_check_equal_to_O_or_1(expected, array):
    run_test_check(check_equal_to_O_or_1, expected, array)


#######

N = 100
tol = 1e-8


@pytest.mark.parametrize(
    "expected, array",
    [
        (True, None),
        (True, np.ones(N)),
        (True, np.zeros(N)),
        (True, -tol * np.ones(N)),
        (True, tol + np.ones(N)),
        (False, -2 * tol * np.ones(N)),
        (False, np.ones(N) + 2 * tol),
        (True, random_array_uniform(N)),
        (True, random_array_uniform(N, N)),
        (True, 1 - random_array_uniform(N)),
        (False, 1 + random_array_uniform(N)),
        (True, -tol + random_array_uniform(N)),
        (True, random_array_uniform(N, N)),
    ],
)
def test_check_in_01(expected, array):
    run_test_check(check_in_01, expected, array)


@pytest.mark.parametrize(
    "expected, array",
    [
        (True, None),
        (True, np.zeros(N)),
        (True, np.ones(N)),
        (False, -np.ones(N)),
        (True, -tol * np.ones(N)),
        (False, -2 * tol * np.ones(N)),
    ],
)
def test_check_geq_0(expected, array):
    run_test_check(check_geq_0, expected, array)
