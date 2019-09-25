# coding: utf8
""" Unit tests:

- :class:`TestRSKCorrespondence` check correct implementation of RSK algorithm
"""


import unittest

import sys
sys.path.append('..')

from dppy.exotic_dpps_core import RSK


class TestRSKCorrespondence(unittest.TestCase):
    """ Test Robinson-Schensted-Knuth correspondence

    Examples come from `Connor Ahlbach's paper <https://pdfs.semanticscholar.org/613e/6105bd02c4f6442d9f255534294230253aa3.pdf>`_ see
    Definition 1.6 and Proposition 2.2
    """

    sequences = [list(range(1, 7)),
                 list(range(6, 0, -1)),
                 [3, 6, 1, 4, 5, 2],
                 [6, 2, 3, 5, 1, 4],
                 [5, 2, 6, 1, 4, 3],
                 [4, 2, 6, 5, 1, 3]]

    PQ_tableaux = [
        ([[1, 2, 3, 4, 5, 6]], [[1, 2, 3, 4, 5, 6]]),
        ([[1], [2], [3], [4], [5], [6]], [[1], [2], [3], [4], [5], [6]]),
        ([[1, 2, 5], [3, 4], [6]], [[1, 2, 5], [3, 4], [6]]),
        ([[1, 3, 4], [2, 5], [6]], [[1, 3, 4], [2, 6], [5]]),
        ([[1, 3], [2, 4], [5, 6]], [[1, 3], [2, 5], [4, 6]]),
        ([[1, 3], [2, 5], [4, 6]], [[1, 3], [2, 4], [5, 6]])]

    def test_RSK_output(self):

        for i, (seq, tab) in enumerate(zip(self.sequences, self.PQ_tableaux)):
            with self.subTest(index=i, seq=seq):
                self.assertTrue(RSK(seq) == tab)


def main():

    unittest.main()


if __name__ == '__main__':
    main()
