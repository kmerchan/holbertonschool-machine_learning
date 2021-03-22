#!/usr/bin/env python3
"""
Defines a function that calculates a correlation matrix
"""


import numpy as np


def correlation(C):
    """
    Calculates a correlation matrix

    parameters:
        C [numpy.ndarray of shape (d, d)]: contains a covariance matrix
            d: number of dimensions

    returns:
        [numpy.ndarray of shape (d, d)]:
            containing the correlation matrix
    """
    if type(C) is not np.ndarray:
        raise TypeError("C must be a numpy.ndarray")
    if len(C.shape) != 2:
        raise ValueError("C must be a 2D square matrix")
    d, d_2 = C.shape
    if d != d_2:
        raise ValueError("C must be a 2D square matrix")
    D = np.sqrt(np.diag(C))
    D_inverse = 1 / np.outer(D, D)
    corr = D_inverse * C
    return corr
