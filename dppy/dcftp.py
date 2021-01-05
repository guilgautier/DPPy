from math import sqrt, log
from numpy.random import poisson
from random import random, sample
from collections import deque
# With the current Python implementation, we use both random and numpy.random and we do not populate the RNG, hence reproducibility is not possible.


def distance(x, y):
    return sqrt((y[0] - x[0])**2 + (y[1] - x[1])**2)

def backward_update(D, X, M, n, beta, w, h):

    b_rate = beta * w * h

    for _ in range(-n, -2 * n, -1):

        card_D = len(D)

        if random() < card_D / (b_rate + card_D):
            x = sample(D, 1)[0]
            D.discard(x)
            X.appendleft(x)
            M.appendleft(random())
        else:
            x = (w * random(), h * random())
            D.add(x)
            X.appendleft(x)
            M.appendleft(0.0)

    return D, X, M

def forward_coupling(D, X, M, gamma, r):

    L, U = set(), D.copy()
    log_g = log(gamma)

    for x, m in zip(X, M):
        if m:
            log_m = log(m)

            neigh_x_U = sum(distance(x, u) <= r for u in U)
            if log_m < neigh_x_U * log_g:
                L.add(x)
                U.add(x)
            else:
                neigh_x_L = sum(distance(x, l) <= r for l in L)
                if log_m < neigh_x_L * log_g:
                    U.add(x)

        else:
            L.discard(x)
            U.discard(x)

    return L, U

def strauss_dcftp(beta, gamma, r, w, h):

    lmbda = poisson(beta * w * h)
    D = set((w * random(), h * random()) for _ in range(lmbda))
    X = deque()
    M = deque()

    n = 1
    while True:
        D, X, M = backward_update(D, X, M, n, beta, w, h)
        L, U = forward_coupling(D, X, M, gamma, r)
        if len(L) == len(U):  # coalescence occurs
            return L, n
        n *= 2


if __name__ == '__main__':

    beta, gamma, r = 2, 0.2, 0.7
    w, h = 10, 10
    config, iter_ = strauss_dcftp(beta, gamma, r, w, h)
    print(len(config), iter_)
