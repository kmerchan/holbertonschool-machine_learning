#!/usr/bin/env python3
"""
Defines function that shuffles the data points
in two matrices the same way
"""


import numpy as np


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way

    parameters:
        X [numpy.ndarray of shape (m, nx)]:
            first matrix to shuffle
            m: number of data points
            nx: the number of features in X
        Y [numpy.ndarray of shape (m, ny)]:
            second matrix to shuffle
            m: number of data points, same as in X
            ny: the number of features in Y

    returns:
        the shuffled X and Y matrices, respectively
    """
    m = X.shape[0]
    shuffle_pattern = np.random.permutation(m)
    X_shuffled = X[shuffle_pattern]
    Y_shuffled = Y[shuffle_pattern]
    return (X_shuffled, Y_shuffled)
