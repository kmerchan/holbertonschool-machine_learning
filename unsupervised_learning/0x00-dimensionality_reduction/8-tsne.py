#!/usr/bin/env python3
"""
Defines function that performs a t-SNE transformation
"""


import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """
    Performs a t-SNE transformation

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            containing the dataset to be transformed by t-SNE
            n: the number of data points
            d: the number of dimensions in each point
        ndims [int]:
            the new dimensional representation of X
        idims [int]:
            the intermediate dimensional representation of X after PCA
        perplexity [float]:
            perplexity that all Gaussian distributions should have
        iterations [int]:
            the number of iterations
        lr [float]:
            the learning rate

    returns:
        Y [numpy.ndarray of shape (n, ndim)]:
            contatining the optimized low dimensional transformation of X
    """
    return None
