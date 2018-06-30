import unittest
import scipy.stats as sps
import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import sys
sys.path.append("..")
from dppy.discrete_dpps import *

# Here's our "unit tests".
class FollowsTheRightMarginal(unittest.TestCase):

    def testGS(self):
        """
        test whether GS samples the right marginals
        """
        npr.seed(1)

        # build a projection matrix K
        N = 10
        A = npr.randn(N**2).reshape((N,N))
        Q, R = npl.qr(A)
        L = np.eye(N)
        r = N//2 # rank of the projection used for testing
        for i in range(N-r):
            L[i,i] = 0
        K = Q.dot(L.dot(Q.T)) # projection matrix by construction
        dpp = Discrete_DPP(kernel_type="inclusion", proj=True, K=K)

        # sample from the dpp
        numSamples = 100
        for _ in range(numSamples):
            dpp.sample_exact(sampling_mode="GS")

        # perform chi-square test
        h, bins = np.histogram([item for sublist in dpp.list_of_samples for item in sublist], bins = np.arange(N+1))
        observedFrequencies = h/np.sum(h)
        expectedFrequencies = np.diag(K)/r
        chi2, pval = sps.chisquare(observedFrequencies, expectedFrequencies)
        self.failUnless(pval>0.05)

def main():
    unittest.main()

if __name__ == '__main__':
    main()
