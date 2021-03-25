#!/usr/bin/env python3
"""
Defines function that performs principal components analysis (PCA) on dataset
"""


import numpy as np


def pca(X, var=0.95):
    """
    Performs principal components analysis (PCA) on a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]: dataset
            n: number of data points
            d: number of dimensions in each data point
            all dimensions have a mean of 0 across all data points
        var [float]: the fraction of the variance that the PCA
            transformation should maintain

    returns:
        W [numpy.ndarray of shape (d, nd)]: the weights matrix that
            maintains var fraction of X's original variance
            d: number of dimensions of each data point
            nd: new dimensionality of the transformed X
    """
    # n, d = X.shape
    u, s, v = np.linalg.svd(X)
    ratios = list(x / np.sum(s) for x in s)
    variance = np.cumsum(ratios)
    nd = np.argwhere(variance >= var)[0, 0]
    W = v.T[:, :(nd + 1)]
    return (W)
