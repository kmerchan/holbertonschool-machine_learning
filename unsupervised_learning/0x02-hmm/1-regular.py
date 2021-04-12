#!/usr/bin/env python3
"""
Defines function that determines the steady state probabilities of
a regular Markov Chain
"""


import numpy as np


def regular(P):
    """
    Determines the steady state probabilities of a regular Markov Chain

    parameters:
        P [square 2D numpy.ndarray of shape (n, n)]:
            representing the transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n: the number of state in the Markov Chain

    returns:
        [a numpy.ndarray of shape (1, n)]:
            representing the steady state probabilities
        or None on failure
    """
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return None
    if not (P > 0).all():
        return None
    Identity = np.identity(n)
    Q = P - Identity
    e = np.ones((n,))
    Qe = np.c_[Q, e]
    QTQ = np.matmul(Qe, Qe.T)
    QbT = np.ones((n,))
    result = np.linalg.solve(QTQ, QbT)
    return np.expand_dims(result, axis=0)
