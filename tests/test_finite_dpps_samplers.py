# coding: utf8
""" Unit tests:

- :class:`TestFiniteDppSamplers`
"""

import unittest

from itertools import chain  # to flatten list of samples

import numpy as np
import numpy.random as rndm

import scipy.linalg as la
from scipy.stats import chisquare

import sys
sys.path.append('..')

from dppy.finite_dpps import FiniteDPP
from dppy.utils import (det_ST,
                        example_eval_L_linear,
                        example_eval_L_min_kern)


class TestAdequationOfFiniteDppSamplers(unittest.TestCase):
    """ For various settings, test that the
    """
    rank, N = 6, 10
    nb_exact_samples = 1000
    nb_iter_mcmc = nb_exact_samples

    e_vals_eq_01 = np.ones(rank)
    e_vals_in_01 = np.random.rand(rank)
    e_vals_geq_0 = 1 + rndm.geometric(p=0.5, size=rank)

    e_vecs, _ = la.qr(rndm.randn(N, rank), mode='economic')

    A_zono = rndm.randn(rank, N)

    phi = rndm.randn(rank, N)

    @staticmethod
    def singleton_adequation(dpp, samples, tol=0.05):
        """Perform chi-square test to check that
        P[{i} C X] = K_ii
        """

        if dpp.size_k_dpp:
            return True, 'We do not check inclusion probabilities for k-DPPs'
        else:
            dpp.compute_K()
            N = len(dpp.K)

            marginal_th = np.diag(dpp.K) / np.trace(dpp.K)

            samples_as_singletons = list(chain.from_iterable(samples))
            marginal_emp, _ = np.histogram(samples_as_singletons,
                                           bins=range(N + 1),
                                           density=True)

            _, pval = chisquare(f_obs=marginal_emp, f_exp=marginal_th)

            msg = 'pval = {}, emp = {}, th = {}'.format(pval,
                                                        marginal_emp,
                                                        marginal_th)

            return pval > tol, msg

    @staticmethod
    def doubleton_adequation(dpp, samples, tol=0.05):
        """Perform chi-square test to check that
        P[{i, j} C X] = det [[K_ii, K_ij], [K_ji, K_jj]]
        """
        if dpp.size_k_dpp:
            return True, 'We do not check inclusion probabilities for k-DPPs'
        else:
            samples = list(map(set, samples))
            dpp.compute_K()
            N = len(dpp.K)

            nb_doubletons = 10
            doubletons = [set(rndm.choice(N, size=2, replace=False))
                          for _ in range(nb_doubletons)]

            # det [[K_ii, K_ij], [K_ji, K_jj]]
            marginal_th = [det_ST(dpp.K, list(d)) for d in doubletons]

            counts = [sum([doubl.issubset(sampl) for sampl in samples])
                      for doubl in doubletons]
            marginal_emp = np.array(counts) / len(samples)

            _, pval = chisquare(f_obs=marginal_emp, f_exp=marginal_th)

            msg = 'pval = {}, emp = {}, th = {}'.format(pval,
                                                        marginal_emp,
                                                        marginal_th)

            return pval > tol, msg

    @staticmethod
    def uniqueness_of_items(dpp, samples):
        """Check that each sample is made of unique items (no duplicates)
        """

        adeq = all(len(set(x)) == len(x) for x in samples)
        msg = 'Some samples contain duplicated items, while each item must appear only once'

        return adeq, msg

    @staticmethod
    def cardinality_adequation(dpp, samples):
        """Check that the empirical cardinality of the samples is within a standard deviation to the true E[|X|] = Trace(K).
        For k-DPP, simply check that the samples have the prescribed cadinality"""

        card_emp = np.array(list(map(len, samples)))
        mean_card_emp = np.mean(card_emp)

        if dpp.size_k_dpp:
            adeq = np.all(card_emp == dpp.size_k_dpp)
            msg = '|X|_emp = {}, |X|_th = {}'.format(mean_card_emp,
                                                     dpp.size_k_dpp)
            return adeq, msg
        else:
            dpp.compute_K()
            # E[|X|] = Trace(K), Var[|X|] = Trace(K - K^2)
            mean_card_th = np.trace(dpp.K)
            if dpp.kernel_type == 'correlation' and dpp.projection:
                # E[|X|] = |X| = Trace(K) |X| = rank(K)
                mean_card_th = np.rint(mean_card_th).astype(int)

            std_card_th = np.sqrt(np.abs(np.trace(dpp.K - dpp.K.dot(dpp.K))))

            adeq = np.abs(mean_card_emp - mean_card_th) <= std_card_th
            msg = 'E_emp = {}, E_th = {}, Std_th = {}'.format(mean_card_emp,
                                                              mean_card_th,
                                                              std_card_th)

            return adeq, msg

    def adequation(self, typ, samples, dpp):
        if typ == 'uniqueness_of_items':
            return self.uniqueness_of_items(dpp, samples)
        elif typ == 'cardinality':
            return self.cardinality_adequation(dpp, samples)
        elif typ == 'singleton':
            return self.singleton_adequation(dpp, samples)
        elif typ == 'doubleton':
            return self.doubleton_adequation(dpp, samples)

    def run_adequation_tests(self,
                             kernel_type,
                             list_dpp_params,
                             dict_sampler_mode_param,
                             adequation_to_check):

        for sampler, modes in dict_sampler_mode_param.items():
            for md, md_params in modes.items():
                for idx, (proj, dpp_param) in enumerate(list_dpp_params):
                    dpp = FiniteDPP(kernel_type, projection=proj, **dpp_param)
                    with self.subTest(idx=idx,
                                      dpp=(dpp.kernel_type,
                                           dpp.projection,
                                           dpp.params_keys),
                                      sampler=(sampler,
                                               md,
                                               md_params)):
                        dpp.flush_samples()
                        samples = self.get_samples(dpp,
                                                   sampler,
                                                   md,
                                                   **md_params)
                        for adeq_typ in adequation_to_check:
                            with self.subTest(adequation=adeq_typ):
                                adeq, msg = self.adequation(adeq_typ,
                                                            samples,
                                                            dpp)
                                self.assertTrue(adeq, msg)

    def get_samples(self, dpp, sampler, mode, **params):

        if sampler.startswith('exact'):
            for _ in range(self.nb_exact_samples):
                if sampler == 'exact_dpp':
                    dpp.sample_exact(mode=mode, **params)
                elif sampler == 'exact_k_dpp':
                    dpp.sample_exact_k_dpp(mode=mode, **params)
                else:
                    raise ValueError(sampler)
            return dpp.list_of_samples

        elif sampler.startswith('mcmc'):
            if sampler == 'mcmc_dpp':
                dpp.sample_mcmc(mode=mode, **params)
            elif sampler == 'mcmc_k_dpp':
                dpp.sample_mcmc_k_dpp(mode=mode, **params)
            else:
                raise ValueError(sampler)
            return dpp.list_of_samples[-1]

    def test_adequation_of_projection_dpp_K_zonotope_sampler(self):

        kernel_type = 'correlation'
        # projection, param
        list_dpp_params = [(True, {'A_zono': self.A_zono})]

        dict_sampler_mode_param = {'mcmc_dpp': {'zonotope': {}}}

        adequation_to_check = ('uniqueness_of_items', 'cardinality', 'singleton', 'doubleton')

        self.run_adequation_tests(kernel_type,
                                  list_dpp_params,
                                  dict_sampler_mode_param,
                                  adequation_to_check)

    def test_dpp_adequation_with_projection_correlation_kernel(self):

        kernel_type = 'correlation'
        # projection, param
        list_dpp_params =\
            [(True,
                {'K': (self.e_vecs*self.e_vals_eq_01).dot(self.e_vecs.T)}),
             (True,
                {'K_eig_dec': (self.e_vals_eq_01, self.e_vecs)}),
             (True,
                {'A_zono': self.A_zono})]

        k = self.rank
        dict_sampler_mode_param =\
            {'exact_dpp': {'GS': {},
                           'Chol': {},
                           'Schur': {}},
             'exact_k_dpp': {'GS': {'size': k}},
             'mcmc_dpp': {'E': {'size': k,
                                'nb_iter': self.nb_iter_mcmc}},
             'mcmc_k_dpp': {'E': {'size': k,
                                  'nb_iter': self.nb_iter_mcmc}}}

        adequation_to_check = ('uniqueness_of_items', 'cardinality', 'singleton', 'doubleton')

        self.run_adequation_tests(kernel_type,
                                  list_dpp_params,
                                  dict_sampler_mode_param,
                                  adequation_to_check)

    def test_dpp_adequation_with_non_projection_correlation_kernel(self):

        kernel_type = 'correlation'
        # projection, param
        list_dpp_params =\
            [(False,
                {'K': (self.e_vecs * self.e_vals_in_01).dot(self.e_vecs.T)}),
             (False,
                {'K_eig_dec': (self.e_vals_in_01, self.e_vecs)})]

        k = self.rank // 2

        dict_sampler_mode_param =\
            {'exact_dpp': {'GS': {},
                           'GS_bis': {},
                           'Chol': {},
                           'KuTa12': {}},
             'exact_k_dpp': {'GS': {'size': k},
                             'GS_bis': {'size': k},
                             'KuTa12': {'size': k}},
             'mcmc_dpp': {'AD': {'nb_iter': self.nb_iter_mcmc},
                          'AED': {'nb_iter': self.nb_iter_mcmc}},
             'mcmc_k_dpp': {'E': {'size': k,
                                  'nb_iter': self.nb_iter_mcmc}}}

        adequation_to_check = ('uniqueness_of_items', 'cardinality', 'singleton', 'doubleton')

        self.run_adequation_tests(kernel_type,
                                  list_dpp_params,
                                  dict_sampler_mode_param,
                                  adequation_to_check)

    def test_dpp_adequation_with_projection_likelihood_kernel(self):

        kernel_type = 'likelihood'
        # projection, param
        list_dpp_params =\
            [(True,
                {'L': (self.e_vecs * self.e_vals_eq_01).dot(self.e_vecs.T)}),
             (True,
                {'L_eig_dec': (self.e_vals_eq_01, self.e_vecs)})]

        k = self.rank
        dict_sampler_mode_param =\
            {'exact_dpp': {'GS': {},
                           'GS_bis': {},
                           'Chol': {},
                           'KuTa12': {}},
             'exact_k_dpp': {'GS': {'size': k}},
             'mcmc_dpp': {'AD': {'nb_iter': self.nb_iter_mcmc},
                          'AED': {'nb_iter': self.nb_iter_mcmc}},
             'mcmc_k_dpp': {'E': {'size': k,
                                  'nb_iter': self.nb_iter_mcmc}}}

        adequation_to_check = ('uniqueness_of_items', 'cardinality', 'singleton', 'doubleton')

        self.run_adequation_tests(kernel_type,
                                  list_dpp_params,
                                  dict_sampler_mode_param,
                                  adequation_to_check)

    def test_dpp_adequation_with_non_projection_likelihood_kernel(self):

        kernel_type = 'likelihood'
        # projection, param
        list_dpp_params =\
            [(False,
                {'L': (self.e_vecs * self.e_vals_eq_01).dot(self.e_vecs.T)}),
             (False,
                {'L_eig_dec': (self.e_vals_eq_01, self.e_vecs)}),
             (False,
                {'L': (self.e_vecs * self.e_vals_geq_0).dot(self.e_vecs.T)}),
             (False,
                {'L_eig_dec': (self.e_vals_geq_0, self.e_vecs)}),
             (False,
                {'L_gram_factor': self.phi})]  # L_gram_factor to test L_dual

        k = self.rank // 2

        dict_sampler_mode_param =\
            {'exact_dpp': {'GS': {},
                           'GS_bis': {},
                           'KuTa12': {}},
             'exact_k_dpp': {'GS': {'size': k},
                             'GS_bis': {'size': k},
                             'KuTa12': {'size': k}},
             'mcmc_dpp': {'AD': {'nb_iter': self.nb_iter_mcmc},
                          'AED': {'nb_iter': self.nb_iter_mcmc}},
             'mcmc_k_dpp': {'E': {'size': k,
                                  'nb_iter': self.nb_iter_mcmc}}}

        adequation_to_check = ('uniqueness_of_items', 'cardinality', 'singleton', 'doubleton')

        self.run_adequation_tests(kernel_type,
                                  list_dpp_params,
                                  dict_sampler_mode_param,
                                  adequation_to_check)

    def test_adequation_vfx_sampler_linear_kernel(self):

        kernel_type = 'likelihood'

        X_data_randn = rndm.rand(100, 6)

        list_dpp_params = [[False, {'L_eval_X_data': (example_eval_L_linear, X_data_randn)}]]

        L_lin = example_eval_L_linear(X_data_randn)
        I_L_lin = L_lin + np.eye(*L_lin.shape)
        exp_card = np.sum(np.diag(L_lin.dot(np.linalg.inv(I_L_lin))))
        k = np.floor(exp_card).astype(int) // 2

        print('E[|X|]={}, k={}'.format(exp_card, k))

        dict_sampler_mode_param =\
            {'exact_dpp': {'vfx': {'verbose': False,
                                   'rls_oversample_bless': 5},
                           'alpha': {'verbose': False,
                                     'rls_oversample_bless': 5},
                           'GS': {}},
             'exact_k_dpp': {'vfx': {'size': k, 'verbose': False,
                                     'rls_oversample_bless': 5},
                             'alpha': {'size': k, 'verbose': False,
                                       'rls_oversample_bless': 5},
                             'GS': {'size': k}}}

        adequation_to_check = ('uniqueness_of_items', 'cardinality', 'singleton', 'doubleton')

        self.run_adequation_tests(kernel_type,
                                  list_dpp_params,
                                  dict_sampler_mode_param,
                                  adequation_to_check)

    def test_adequation_vfx_sampler_min_kernel(self):

        kernel_type = 'likelihood'

        X_data_in_01 = rndm.rand(100, 1)

        list_dpp_params = [[False, {'L_eval_X_data': (example_eval_L_min_kern, X_data_in_01)}]]

        L_min = example_eval_L_min_kern(X_data_in_01)
        I_L_min = L_min + np.eye(*L_min.shape)
        exp_card = np.sum(np.diag(L_min.dot(np.linalg.inv(I_L_min))))
        k = np.floor(exp_card).astype(int) // 2

        print('E[|X|]={}, k={}'.format(exp_card, k))

        dict_sampler_mode_param =\
            {'exact_dpp': {'vfx': {'verbose': False,
                                   'rls_oversample_bless': 10},
                           'alpha': {'verbose': False,
                                     'rls_oversample_bless': 5},
                           'GS': {}},
             'exact_k_dpp': {'vfx': {'size': k, 'verbose': False,
                                     'rls_oversample_bless': 10},
                             'alpha': {'size': k, 'verbose': False,
                                       'rls_oversample_bless': 5},
                             'GS': {'size': k}}}

        adequation_to_check = ('uniqueness_of_items', 'cardinality', 'singleton', 'doubleton')

        self.run_adequation_tests(kernel_type,
                                  list_dpp_params,
                                  dict_sampler_mode_param,
                                  adequation_to_check)


def main():

    unittest.main()


if __name__ == '__main__':
    main()
