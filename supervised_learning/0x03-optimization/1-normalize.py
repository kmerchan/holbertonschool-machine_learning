#!/usr/bin/env python3
"""
Defines function that normalizes a matrix
"""


def normalize(X, m, s):
    """
    Normalizes a matrix

    parameters:
        X [numpy.ndarray of shape (d, nx)]:
            matrix to normalize
            d: number of data points
            nx: the number of features
        m [numpy.ndarray of shape (nx,)]:
            contains the mean of all features of X
        s [numpy.ndarray of shape (nx,)]:
            contains the standard deviation of all
                features of X

    normalization:
        subtract mean:
            mu = (1 / m) * sum of all values from 1 to m
            value = value - mu
        normalize variance after subtracting mean:
            sigma squared = (1 / m) * sum of all values squared
            value = value / sigma squared

    returns:
        the normalized X matrix
    """
    normalized = (X - m) / s
    return (normalized)
