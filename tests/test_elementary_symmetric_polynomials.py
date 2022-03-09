from itertools import combinations

import numpy as np

from dppy.utils import elementary_symmetric_polynomials


def test_elementary_symmetric_polynomials():
    """Test the recursive evaluations of the elementary symmetric polynomials used for sampling k-DPPs

    .. math::

        e_k(\\lambda{1:r}=1, \\lambda{r+1:N}=0)
        = \\sum_{\\substack{S \\subset [N]\\\\|S|=k}}
            \\prod_{s\\in S} \\lambda{s}
    """
    n = 10

    x = np.random.randn(n)
    expected = elementary_symmetric_polynomials(n, x)

    computed = np.zeros((n + 1, n + 1))
    computed[0, :] = 1.0
    for k in range(1, n + 1):
        for n in range(1, n + 1):
            computed[k, n] = sum(np.prod(y) for y in combinations(x[:n], k))

    np.testing.assert_array_almost_equal(computed, expected)
