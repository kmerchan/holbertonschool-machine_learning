#!/usr/bin/env python3
"""
Defines function that calculates the symmetric P affinities
"""


import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """
    Calculates the symmetric P affinities of a data set

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            containing the dataset to be transformed by t-SNE
            n: the number of data points
            d: the number of dimensions in each point
        tol [float]:
            maximum tolerance allowed (inclusive) for the difference in
                Shannon entropy from perplexity for all Gaussian distributions
        perplexity:
            perplexity that all Gaussian distributions should have

    returns:
        P [numpy.ndarray of shape (n, n)]:
            contatining the symmetric P affinities
    """
    return None
