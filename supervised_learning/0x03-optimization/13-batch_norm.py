#!/usr/bin/env python3
"""
Defines function that normalizes an unactivated output of a neural network
using batch normalization
"""


def batch_norm(Z, gamma, beta, epsilon):
    """
    Normalizes an unactivated output of a neural network
        using batch normalization

    parameters:
        Z [numpy.ndarray of shape (m, n)]:
            unactivated output to normalize
            m: number of data points
            n: number of features in Z
        gamma [numpy.ndarray of shape (1, n)]:
            contains the scales used for batch normalization
        beta [numpy.ndarray of shape (1, n)]:
            contains the offsets used for batch normalization
        epsilon [float]: small number to avoid division by zero

    returns:
        the normalized Z matrix
    """
