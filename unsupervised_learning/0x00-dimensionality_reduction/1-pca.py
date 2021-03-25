#!/usr/bin/env python3
"""
Defines function that performs principal components analysis (PCA) on dataset
"""


import numpy as np


def pca(X, ndim):
    """
    Performs principal components analysis (PCA) on a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]: dataset
            n: number of data points
            d: number of dimensions in each data point
        ndim [int]: the new dimensionality of the transformed X

    returns:
        T [numpy.ndarray of shape (n, ndim)]:
            containing the transformed version of X
            n: number of data points
            ndim: the new dimensionality of the transformed X
    """
    # n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    A = X - mean
    u, s, v = np.linalg.svd(A)
    W = v.T[:, :ndim]
    T = np.matmul(A, W)
    return (T)
