#!/usr/bin/env python3
"""
Defines function that calculates the
normalization constants of a matrix
"""


import numpy as np


def normalization_constants(X):
    """
    Calculates the normalization constants of a matrix

    parameters:
        X [numpy.ndarray of shape (m, nx)]:
            matrix to find normalization constants for
            m: number of data points
            nx: the number of features

    normalization:
        subtract mean:
            mu = (1 / m) * sum of all values from 1 to m
            value = value - mu
        normalize variance after subtracting mean:
            sigma squared = (1 / m) * sum of all values squared
            value = value / sigma squared

    returns:
        the mean and standard deviation of each feature, respectively
    """
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    return (mu, sigma)
