# coding: utf8
""" Tests:

Cheap test to check that the ploting functions output something
Use runpy to run the plots displayed in the documentation, instead of rewriting everything!
"""

import unittest
import os
import runpy

import matplotlib.pyplot as plt

import sys
sys.path.append('..')  # make sure dppy is available when run plot files

class TestPlot(unittest.TestCase):

    dir_of_this_test = os.path.dirname(os.path.realpath(__file__))
    dir_plots_to_test = dir_of_this_test.replace('tests', 'docs/plots')

    def test_plot(self):

        list_plot_files = ['/'.join([self.dir_plots_to_test, f])
                           for f in os.listdir(self.dir_plots_to_test)
                           if f.startswith('ex_plot') and f.endswith('.py')]

        for path in list_plot_files:
            with self.subTest(path_to_plot_example=path):

                plt.close()
                nb_fig_before = plt.gcf().number
                runpy.run_path(path)
                nb_fig_after = plt.gcf().number

                self.assertTrue(nb_fig_after > nb_fig_before)


def main():

    unittest.main()


if __name__ == '__main__':
    main()
