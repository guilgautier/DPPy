# coding: utf8
""" Unit tests:

- :class:`TestBless` check correct implementation of the corresponding algorithm.
"""

import unittest
import numpy as np
import numpy.linalg as la

import sys
sys.path.append('..')

from dppy.bless import (bless,
                        estimate_rls_bless,
                        reduce_lambda,
                        CentersDictionary)

from dppy.utils import (check_random_state,
                        example_eval_L_polynomial,
                        evaluate_L_diagonal)


class TestBless(unittest.TestCase):
    """
    """

    seed = 42

    def __create_lambda_acc_dictionary(self, X_data, eval_L, lam, rng):
        n = X_data.shape[0]
        rls_oversample = 5
        diag_norm = np.asarray(evaluate_L_diagonal(eval_L, X_data))
        init_rls_estimate = diag_norm / (diag_norm + lam)

        selected_init = rng.rand(n) <= rls_oversample * init_rls_estimate

        # force at least one sample to be selected
        selected_init[0] = 1

        return CentersDictionary(idx=selected_init.nonzero()[0],
                                 X=X_data[selected_init, :],
                                 probs=np.ones(np.sum(selected_init)) * init_rls_estimate[selected_init],
                                 lam=n,
                                 rls_oversample=rls_oversample)

    def test_estimate_rls_bless(self):
        N, d = 100, 5
        lam = 11
        lam_new = 10
        rng = check_random_state(self.seed)
        X_data = rng.randn(N, d)
        L_data = example_eval_L_polynomial(X_data)

        rls_exact = la.solve(L_data + lam_new * np.eye(N), L_data).diagonal()

        dict_exact = self.__create_lambda_acc_dictionary(X_data, example_eval_L_polynomial, 0.0, rng)
        dict_approx = self.__create_lambda_acc_dictionary(X_data, example_eval_L_polynomial, lam, rng)

        rls_estimates_exact = estimate_rls_bless(dict_exact, X_data, example_eval_L_polynomial, lam_new)
        rls_estimates_approx = estimate_rls_bless(dict_approx, X_data, example_eval_L_polynomial, lam_new)


        np.testing.assert_almost_equal(rls_estimates_exact, rls_exact)
        np.testing.assert_allclose(rls_estimates_approx, rls_exact, rtol=0.5)

    def test_reduce_lambda(self):
        N, d = 100, 5
        lam = 11
        lam_new = 10
        rng = check_random_state(self.seed)
        X_data = rng.randn(N, d)
        L_data = example_eval_L_polynomial(X_data)

        rls_exact = la.solve(L_data + lam * np.eye(N), L_data).diagonal()
        dict_approx = self.__create_lambda_acc_dictionary(X_data, example_eval_L_polynomial, lam, rng)
        rls_estimates = estimate_rls_bless(dict_approx, X_data, example_eval_L_polynomial, lam)
        np.testing.assert_allclose(rls_estimates,
                                   rls_exact,
                                   rtol=0.5)

        dict_reduced = reduce_lambda(X_data, example_eval_L_polynomial, dict_approx, lam_new, rng)

        rls_estimates_reduced = estimate_rls_bless(dict_reduced, X_data, example_eval_L_polynomial, lam_new)
        rls_exact_reduced = la.solve(L_data + lam_new * np.eye(N), L_data).diagonal()

        np.testing.assert_allclose(rls_estimates_reduced,
                                   rls_exact_reduced,
                                   rtol=0.5)

        self.assertTrue(len(dict_reduced.idx) <= len(dict_approx.idx))

    def test_bless(self):
        N, d = 100, 5
        lam = 11
        rng = check_random_state(self.seed)
        X_data = rng.randn(N, d)
        L_data = example_eval_L_polynomial(X_data)

        rls_exact = la.solve(L_data + lam * np.eye(N), L_data).diagonal()

        dict_reduced = bless(X_data,
                             example_eval_L_polynomial,
                             lam_final=lam,
                             rls_oversample_param=5,
                             random_state=rng,
                             verbose=False)

        rls_estimates = estimate_rls_bless(dict_reduced,
                                           X_data,
                                           example_eval_L_polynomial,
                                           lam)

        np.testing.assert_allclose(rls_estimates, rls_exact, rtol=0.5)

        self.assertTrue(len(dict_reduced.idx) <= 5 * rls_exact.sum())


def main():

    unittest.main()


if __name__ == '__main__':
    main()
