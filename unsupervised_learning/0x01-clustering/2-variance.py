#!/usr/bin/env python3
"""
Defines function that calculates total intra-cluster variance for a data set
"""


import numpy as np


def variance(X, C):
    """
    Calculates the total intra-cluster variance for a data set

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        C [numpy.ndarray of shape (k, d)]:
            contains the centroid means for each cluster
            k: the number of clusters
            d: the number of dimensions for each data point

    should not use any loops

    returns:
        var [float]: total variance
        or None on failure
    """
    return None
