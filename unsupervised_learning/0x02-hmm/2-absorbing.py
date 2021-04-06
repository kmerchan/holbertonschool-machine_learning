#!/usr/bin/env python3
"""
Defines function that determines if the Markov Chain is absorbing
"""


import numpy as np


def absorbing(P):
    """
    Determines if the Markov Chain is absorbing

    parameters:
        P [square 2D numpy.ndarray of shape (n, n)]:
            representing the standard transition matrix
            P[i, j] is the probability of transitioning from state i to state j
            n: the number of state in the Markov Chain

    returns:
        True, if absorbing
        False, if not absorbing or on failure
    """
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return False
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return False
    return True
