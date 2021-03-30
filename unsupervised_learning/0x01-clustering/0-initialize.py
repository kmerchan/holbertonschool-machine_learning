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

    should not use any loops

    returns:
        [numpy.ndarray of shape (k, d)]:
            containing the initialized centroids for each cluster
        or None on failure
    """
    return None
