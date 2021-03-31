#!/usr/bin/env python3
"""
Defines function that tests for the optimum number of clusters by variance
"""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    Tests for the optimum number of clusters by variance

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        kmin [positive int]:
            containing the minimum number of clusters to check for (inclusive)
        kmax [positive int]:
            containing the maximum number of clusters to check for (inclusive)
        iterations [positive int]:
            containing the maximum number of iterations for K-means

    function should analyze at least 2 different cluster sizes

    should use at most 2 loops

    returns:
        results, d_vars:
            results [list]:
                containing the output of K-means for each cluster size
            d_vars [list]:
                containing the difference in variance from the smallest cluster
                    size for each cluster size
        or None, None on failure
    """
    return None, None
