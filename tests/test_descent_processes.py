import unittest

import numpy as np

from dppy.descent_processes import CarriesProcess, DescentProcess, VirtualDescentProcess
from dppy.utils import check_random_state


class TestDescentProcesses(unittest.TestCase):
    """Check that the marginal probability of selecting any integer is close to the theoretical ``marginal_descent_probability``"""

    size = 50_000
    tol = 1e-2

    def marginal_adequation(self, dpp):
        rng = check_random_state(None)
        sample = dpp.sample(size=self.size, random_state=rng)
        p_hat = np.mean(sample[1:])
        p_th = dpp.marginal_descent_probability

        self.assertTrue(
            np.abs(p_hat - p_th) / p_th < self.tol,
            "p_hat={}, p_th={}".format(p_hat, p_th),
        )

    def test_carries_process(self):
        dpp = CarriesProcess(base=10)
        self.marginal_adequation(dpp)

    def test_descent_process(self):
        dpp = DescentProcess()
        self.marginal_adequation(dpp)

    def test_virtual_descent_process(self):
        dpp = VirtualDescentProcess(x0=0.5)
        self.marginal_adequation(dpp)
