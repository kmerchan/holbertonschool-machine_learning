#!/usr/bin/env python3
"""
Defines function that calculates the probability density function of
a Gaussian distribution
"""


import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset whose PDF should be calculated
            n: the number of data points
            d: the number of dimensions for each data point
        m [numpy.ndarray of shape (d,)]:
            contains the mean of the distribution
        S [numpy.ndarray of shape (d, d)]:
            contains the covariance of the distribution

    not allowed to use any loops
    not allowed to use the function numpy.diag or method numpy.ndarray.diagonal

    returns:
        P [numpy.ndarray of shape (n,)]:
            containing the PDF values for each data point
            all values in P should have a minimum value of 1e-300
        or None on failure
    """
    return None
