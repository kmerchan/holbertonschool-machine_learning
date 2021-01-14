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
    nx = X.shape[1]
    ny = Y.shape[1]
    shuffle_pattern = np.random.permutation(m)
    X_shuffled = np.zeros((m, nx)).astype("int")
    Y_shuffled = np.zeros((m, ny)).astype("int")
    for i in range(len(shuffle_pattern)):
        for x in range(nx):
            X_shuffled[i][x] += X[shuffle_pattern[i]][x]
        for y in range(ny):
            Y_shuffled[i][y] += Y[shuffle_pattern[i]][y]
    return (X_shuffled, Y_shuffled)
