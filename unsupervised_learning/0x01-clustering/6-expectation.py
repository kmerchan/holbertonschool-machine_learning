#!/usr/bin/env python3
"""
Defines function that calculates the expectation step in the EM algorithm
for a Gaussian Mixture Model
"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        pi [numpy.ndarray of shape (k,)]:
            contains the priors for each cluster
        m [numpy.ndarray of shape (k, d)]:
            contains the centroid means for each clustern
        S [numpy.ndarray of shape (k, d, d)]:
            contains the covariance matrices for each cluster

    should only use one loop

    returns:
        g, l:
            g [numpy.ndarray of shape (k, n)]:
                containing the posterior probabilities for each data point
                    in the cluster
            l [float]:
                total log likelihood
        or None, None on failure
    """
    return None, None
