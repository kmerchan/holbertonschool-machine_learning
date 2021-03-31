#!/usr/bin/env python3
"""
Defines function that finds the best number of clusters for a GMM using
the Bayesian Information Criterion (BIC)
"""


import numpy as np
expectation_maximization = __import__('7-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """
    Find the best number of clusters for a GMM using BIC

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        kmin [positive int]:
            the minimum number of clusters to check for (inclusive)
        kmax [positive int]:
            the maximum number of clusters to check for (inclusive)
            if None, kmax should be set to maximum number of clusters possible
        iterations [positive int]:
            the maximum number of iterations for the algorithm
        tol [non-negative float]:
            the tolerance of the log likelihood, used for early stopping
        verbose [boolean]:
            determines if you should print information about the algorithm

    should only use one loop

    returns:
        best_k, best_result, l, b
            best_k [positive int]:
                the best value for k based on its BIC
            best_result [tuple containing pi, m, S]:
                pi [numpy.ndarray of shape (k,)]:
                    contains cluster priors for the best number of clusters
                m [numpy.ndarray of shape (k, d)]:
                    contains centroid means for the best number of clusters
                S [numpy.ndarray of shape (k, d, d)]:
                    contains covariance matrices for best number of clusters
            l [numpy.ndarray of shape (kmax - kmin + 1)]:
                contains the log likelihood for each cluster size tested
            b [numpy.ndarray of shape (kmax - kmin + 1)]:
                contains the BIC value for each cluster size tested
                BIC = p * ln(n) - 2 * 1
                    p: number of parameters required for the model
                    n: number of data points used to create the model
                    l: the log likelihood of the model
        or None, None, None, None on failure
    """
    return None, None, None, None
