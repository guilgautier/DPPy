# coding: utf8
""" Tests:

Cheap test to check that the ploting functions output something
Use runpy to run the plots displayed in the documentation, instead of rewriting everything!
"""

import unittest
import os
import runpy

import matplotlib.pyplot as plt


class TestPlot(unittest.TestCase):

    dir_tests = os.path.dirname(os.path.realpath(__file__))
    dir_tests = dir_tests.replace('tests', 'docs/plots')

    def test_plot(self):

        test_plot_files = ['/'.join([self.dir_tests, f])
                           for f in os.listdir(self.dir_tests)
                           if f.startswith('ex_plot') and f.endswith('.py')]

        for f in test_plot_files:
            print(f)

            nb_fig_before = plt.gcf().number
            runpy.run_path(f)
            nb_fig_after = plt.gcf().number

            self.assertTrue(nb_fig_after > nb_fig_before)
