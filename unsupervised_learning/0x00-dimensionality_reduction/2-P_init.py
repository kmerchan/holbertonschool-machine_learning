#!/usr/bin/env python3
"""
Defines function that initializes all variables required to
calculate the P affinities in t-SNE
"""


import numpy as np


def P_init(X, perplexity):
    """
    Initializes all variables required to
    calculate the P affinities in t-SNE

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            containing the dataset to be transformed by t-SNE
            n: the number of data points
            d: the number of dimensions in each point
        perplexity:
            perplexity that all Gaussian distributions should have

    returns:
        (D, P, betas, H)
        D [numpy.ndarray of shape (n, n)]:
            calculates the squared pairwise distance between two data points
            The diagonal D should be 0s
        P [numpy.ndarray of shape (n, n)]:
            initialized to all 0s that will contain P affinities
        betas [numpy.ndarray of shape (n, 1)]:
            initialized to all 1s that will contain all the beta values
        H: the Shannon entropy for perplexity with a base of 2
    """
    n = X.shape[0]
    mult = np.matmul(X, -X.T)
    summation = np.sum(np.square(X), 1)
    D = np.add(np.add(2 * mult, summation), summation.T)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return (D, P, betas, H)
