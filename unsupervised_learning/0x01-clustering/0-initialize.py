#!/usr/bin/env python3
"""
Defines function that initializes cluster centroids for K-means
"""


import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset that will be used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            contains the number of clusters

    cluster centroids initialized with a multivariate uniform distribution
        along each dimension in d:
        - minimum values for distribution should be the min values of X
            along each dimension in d
        - maximum values for distribution should be the max values of X
            along each dimension in d
        - should only use numpy.random.uniform exactly once

    returns:
        [numpy.ndarray of shape (k, d)]:
            containing the initialized centroids for each cluster
        or None on failure
    """
    # type checks to catch failure
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    # min values of X along each dimension in d
    low = np.min(X, axis=0)
    # max values of X along each dimension in d
    high = np.max(X, axis=0)
    # initialize cluster centroids with multivariate uniform distribution
    centroids = np.random.uniform(low, high, size=(k, d))
    return centroids
