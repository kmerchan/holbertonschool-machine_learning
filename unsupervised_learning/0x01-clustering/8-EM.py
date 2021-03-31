#!/usr/bin/env python3
"""
Defines function that perfoms the expectation maximization (EM)
for a Gaussian Mixture Model
"""


import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """
    Performs the expectation maximization (EM) for a GMM

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            the number of clusters
        iterations [positive int]:
            the maximum number of iterations for the algorithm
        tol [non-negative float]:
            the tolerance of the log likelihood, used for early stopping
            if the difference is less than or equal to tol, stop the algorithm
        verbose [boolean]:
            determines if you should print information about the algorithm
            if true: print 'Log Likelihood after {i} iterations: {l}'
                every 10 iterations and after the last iteration
            {i}: number of iterations of the EM algorithm
            {l}: log likelihood, rounded to 5 decimal places

    should only use one loop

    returns:
        pi, m, S, g, l:
            pi [numpy.ndarray of shape (k,)]:
                containing the priors for each cluster
            m [numpy.ndarray of shape (k, d)]:
                containing the centroid means for each cluster
            S [numpy.ndarray of shape (k, d, d)]:
                containing the covariance matrices for each cluster
            g [numpy.ndarray of shape (k, n)]:
                containing probabilities for each data point in each cluster
            l [float]:
                log likelihood of the model
        or None, None, None, None, None on failure
    """
    return None, None, None, None, None
