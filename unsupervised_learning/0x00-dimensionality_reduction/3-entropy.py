#!/usr/bin/env python3
"""
Defines function that calculates the Shannon entropy and P affinities
relative to a data point
"""


import numpy as np


def HP(Di, beta):
    """
    Calculates the Shannon entropy and P affinities relative to a data point

    parameters:
        Di [numpy.ndarray of shape (n - 1,)]:
            containing the pairwise distances between
                a data point and all other points
            n: the number of data points
        beta [numpy.ndarray of shape (1,)]:
            containing the beta value for the Gaussian distribution

    returns:
        (Hi, Pi)
        Hi: the Shannon entropy of the points
        Pi [numpy.ndarray of shape (n - 1,)]:
            contatining the P affinities of the points
    """
    prob = np.exp(-Di * beta)
    total = np.sum(prob)
    Pi = prob / total
    Hi = -np.sum(Pi * np.log2(Pi))
    return (Hi, Pi)
