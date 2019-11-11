# coding: utf8
""" Unit tests:

- :class:`FiniteDppWithCorrelationKernel`
- :class:`FiniteDppWithLikelihoodKernel`

check that, in various settings, the instanciation of FiniteDPP works well, the computation of the correlation/likelihood kernels is correct
"""

import unittest
import warnings

import numpy as np
import numpy.random as rndm

import scipy.linalg as la

import sys
sys.path.append('..')

from dppy.finite_dpps import FiniteDPP


class FiniteDppWithCorrelationKernel(unittest.TestCase):
    """ Tests on :py:class:`~dppy.finite_dpps.FiniteDPP` defined through its correlation kernel :math:`\\mathbf{K}`, which must satisfy :math:`0 \\preceq K \\preceq I`
    """

    rank, N = 6, 10
    nb_samples = 1000

    e_vals_eq_01 = np.ones(rank)
    e_vals_in_01 = np.random.rand(rank)

    e_vecs, _ = la.qr(rndm.randn(N, rank), mode='economic')

    A_zono = rndm.randn(rank, N)

    # projection, param
    list_of_valid_params =\
        [(True, {'K': (e_vecs * e_vals_eq_01).dot(e_vecs.T)}),
         (True, {'K_eig_dec': (e_vals_eq_01, e_vecs)}),
         (True, {'A_zono': A_zono}),
         (False, {'K': (e_vecs * e_vals_eq_01).dot(e_vecs.T)}),
         (False, {'K_eig_dec': (e_vals_eq_01, e_vecs)}),
         (False, {'A_zono': A_zono}),
         (False, {'K': (e_vecs * e_vals_in_01).dot(e_vecs.T)}),
         (False, {'K_eig_dec': (e_vals_in_01, e_vecs)})]

    def test_instanciation_from_valid_parameters(self):

        for idx, (proj, param) in enumerate(self.list_of_valid_params):
            with self.subTest(index=idx, projection=proj, param=param.keys()):

                self.assertIsInstance(FiniteDPP('correlation', proj, **param),
                                      FiniteDPP)

    def test_instanciation_from_A_zono_with_projection_false_should_raise_warning(self):
        kernel_type, projection = 'correlation', False
        dpp_param = {'A_zono': self.A_zono}

        # raise warning when projection not set to True
        # https://docs.python.org/3/library/warnings.html
        with warnings.catch_warnings(record=True) as w:
            FiniteDPP(kernel_type, projection,
                      **dpp_param)

        self.assertIn('Weird setting', str(w[-1].message))

    def test_instanciation_from_invalid_parameter_key(self):
        kernel_type, projection = 'correlation', False

        for key in ['L', 'L_eig_dec', 'L_gram_factor', 'random_key']:
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as context:
                    FiniteDPP(kernel_type, projection,
                              **{key: (0, 0)})

                self.assertIn('Invalid param', str(context.exception))

    def test_computation_of_correlation_kernel_from_valid_parameters(self):
        kernel_type = 'correlation'

        for idx, (proj, param) in enumerate(self.list_of_valid_params):
            with self.subTest(index=idx, projection=proj, param=param.keys()):

                dpp = FiniteDPP(kernel_type, projection=proj, **param)
                dpp.compute_K()

                if 'K' in param:
                    K = param['K']
                elif 'K_eig_dec' in param:
                    e_vals, e_vecs = param['K_eig_dec']
                    K = (e_vecs * e_vals).dot(e_vecs.T)
                elif 'A_zono' in param:
                    e_vals = np.ones(param['A_zono'].shape[0])
                    e_vecs, _ = la.qr(param['A_zono'].T, mode='economic')
                    K = (e_vecs * e_vals).dot(e_vecs.T)

                self.assertTrue(np.allclose(dpp.K, K))

    def test_computation_of_likehood_kernel_from_valid_parameters(self):
        kernel_type = 'correlation'

        for idx, (proj, param) in enumerate(self.list_of_valid_params):
            with self.subTest(index=idx, projection=proj, param=param.keys()):

                dpp = FiniteDPP(kernel_type, projection=proj, **param)
                try:
                    dpp.compute_L()

                    if 'K' in param:
                        e_vals, e_vecs = la.eigh(param['K'])
                    elif 'K_eig_dec' in param:
                        e_vals, e_vecs = param['K_eig_dec']
                    elif 'A_zono' in param:
                        e_vals = np.ones(param['A_zono'].shape[0])
                        e_vecs, _ = la.qr(param['A_zono'], mode='economic')

                    L = (e_vecs * (e_vals / (1.0 - e_vals))).dot(e_vecs.T)

                    self.assertTrue(np.allclose(dpp.L, L))

                except (ValueError, FloatingPointError) as e:
                    self.assertIn('cannot be computed', str(e.args))


class FiniteDppWithLikelihoodKernel(unittest.TestCase):
    """ Tests on :py:class:`~dppy.finite_dpps.FiniteDPP` defined through its likelihood kernel :math:`\\mathbf{K}`, which must satisfy :math:`\\mathbf{L} \\succeq 0`
    """

    rank, N = 6, 10

    e_vals_eq_01 = np.ones(rank)
    e_vals_geq_0 = 1 + rndm.geometric(p=0.5, size=rank)

    e_vecs, _ = la.qr(rndm.randn(N, rank), mode='economic')

    phi = rndm.randn(rank, N)

    def eval_L_linear(X, Y=None):
        if Y is None:
            return X.dot(X.T)
        else:
            return X.dot(Y.T)

    def eval_L_min(X, Y=None):

        X = np.atleast_2d(X)
        assert X.shape[1] == 1 and np.all((0 <= X) & (X <= 1))

        if Y is None:
            Y = X
        else:
            Y = np.atleast_2d(Y)
            assert Y.shape[1] == 1 and np.all((0 <= Y) & (Y <= 1))

        return np.minimum(np.repeat(X, Y.size, axis=1),
                          np.repeat(Y.T, X.size, axis=0))

    X_data_randn = rndm.rand(N, rank)
    X_data_in_01 = rndm.rand(N, 1)

    list_of_valid_params =\
        [[True, {'L': (e_vecs * e_vals_eq_01).dot(e_vecs.T)}],
         [True, {'L_eig_dec': (e_vals_eq_01, e_vecs)}],
         [False, {'L': (e_vecs * e_vals_eq_01).dot(e_vecs.T)}],
         [False, {'L_eig_dec': (e_vals_eq_01, e_vecs)}],
         [False, {'L': (e_vecs * e_vals_geq_0).dot(e_vecs.T)}],
         [False, {'L_eig_dec': (e_vals_geq_0, e_vecs)}],
         [False, {'L_gram_factor': phi}],
         [False, {'L_gram_factor': phi.T}],
         [False, {'L_eval_X_data': (eval_L_linear, X_data_randn)}],
         [False, {'L_eval_X_data': (eval_L_min, X_data_in_01)}]]

    def test_instanciation_from_valid_parameters(self):

        for idx, (proj, param) in enumerate(self.list_of_valid_params):
            with self.subTest(index=idx, projection=proj, param=param.keys()):

                self.assertIsInstance(FiniteDPP('likelihood', proj, **param),
                                      FiniteDPP)

    def test_instanciation_with_projection_true_should_raise_warning(self):

        # raise warning when projection not set to True
        # https://docs.python.org/3/library/warnings.html
        with warnings.catch_warnings(record=True) as w:
            FiniteDPP(kernel_type='likelihood',
                      projection=True,
                      **{'L_eig_dec': (self.e_vals_eq_01, self.e_vecs)})

        self.assertIn('Weird setting', str(w[-1].message))

    def test_instanciation_from_invalid_parameter_key(self):
        kernel_type, projection = 'likelihood', False

        for key in ['K', 'K_eig_dec', 'A_zono', 'random_key']:
            with self.subTest(key=key):
                with self.assertRaises(ValueError) as context:
                    FiniteDPP(kernel_type, projection,
                              **{key: 0})

                self.assertIn('Invalid param', str(context.exception))

    def test_computation_of_likehood_kernel_from_valid_parameters(self):
        kernel_type = 'likelihood'

        for idx, (proj, param) in enumerate(self.list_of_valid_params):
            with self.subTest(index=idx, projection=proj, param=param.keys()):

                dpp = FiniteDPP(kernel_type, projection=proj, **param)
                dpp.compute_L()

                if 'L' in param:
                    L = param['L']
                elif 'L_eig_dec' in param:
                    e_vals, e_vecs = param['L_eig_dec']
                    L = (e_vecs * e_vals).dot(e_vecs.T)
                elif 'L_gram_factor' in param:
                    L = param['L_gram_factor'].T.dot(param['L_gram_factor'])
                elif 'L_eval_X_data' in param:
                    eval_L, X_data = param['L_eval_X_data']
                    L = eval_L(X_data)

                self.assertTrue(np.allclose(dpp.L, L))

    def test_computation_of_likehood_kernel_should_raise_warning_with_L_eval(self):

        def eval_L_linear(X, Y=None):
            if Y is None:
                return X.dot(X.T)
            else:
                return X.dot(Y.T)

        X_data = rndm.rand(100, 6)

        dpp = FiniteDPP(kernel_type='likelihood',
                        projection=False,
                        **{'L_eval_X_data': (eval_L_linear, X_data)})

        # raise warning when projection not set to True
        # https://docs.python.org/3/library/warnings.html
        with warnings.catch_warnings(record=True) as w:
            dpp.compute_L()

        self.assertIn('Weird setting', str(w[-1].message))

    def test_computation_of_correlation_kernel_from_valid_parameters(self):
        kernel_type = 'likelihood'

        for idx, (proj, param) in enumerate(self.list_of_valid_params):
            with self.subTest(index=idx, projection=proj, param=param.keys()):

                dpp = FiniteDPP(kernel_type, projection=proj, **param)
                dpp.compute_K()

                if 'L' in param:
                    e_vals, e_vecs = la.eigh(param['L'])
                elif 'L_eig_dec' in param:
                    e_vals, e_vecs = param['L_eig_dec']
                elif 'L_gram_factor' in param:
                    L = param['L_gram_factor'].T.dot(param['L_gram_factor'])
                    e_vals, e_vecs = la.eigh(L)
                elif 'L_eval_X_data' in param:
                    eval_L, X_data = param['L_eval_X_data']
                    L = eval_L(X_data)
                    e_vals, e_vecs = la.eigh(L)

                K = (e_vecs * (e_vals / (1.0 + e_vals))).dot(e_vecs.T)

                self.assertTrue(np.allclose(dpp.K, K))


def main():

    unittest.main()


if __name__ == '__main__':
    main()
