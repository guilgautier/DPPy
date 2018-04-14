# coding: utf-8

import numpy as np

# Carries process
def carries_process(N, b=10):

    A = np.random.randint(0,b-1,N)
    B = np.mod(np.cumsum(A),b)

    X = np.zeros((N,), dtype=int)
    X[1:] = B[1:]<B[:-1]

    return X

######################
### Spanning trees ###
######################

# Aldous

# Wilson