#!/usr/bin/env python3
"""
Defines a function that calculates the mean and covariance of a data set
"""


import numpy as np


def mean_cov(X):
    """
    Calculates the mean and covariance of a data set

    parameters:
        X [numpy.ndarray of shape (n, d)]: contains the data set
            n: number of data points
            d: number of dimensions in each data point

    not allowed to use numpy.cov

    returns:
        mean, cov [tuple]:
            mean [numpy.ndarray of shape (1, d)]:
                containing the mean of the data set
            cov [numpy.ndarray of shape (d, d)]:
                containing the covariance matrix of the data set
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    n, d = X.shape
    if n < 2:
        raise ValueError("X must contain multiple data points")
    mean = np.mean(X, axis=0, keepdims=True)
    cov = (1 / (n - 1)) * np.matmul(X.T - mean.T, X - mean)
    return (mean, cov)
