#!/usr/bin/env python3
"""
Defines function that performs K-means on a dataset with sklearn
"""


import sklearn.cluster


def kmeans(X, k):
    """
    Performs K-means on a dataset

    parameters:
        X [numpy.ndarray of shape (n, d)]:
            contains the dataset that will be used for K-means clustering
            n: the number of data points
            d: the number of dimensions for each data point
        k [positive int]:
            contains the number of clusters

    returns:
        C, clss:
            C [numpy.ndarray of shape (k, d)]:
                containing the centroid means for each cluster
            clss [numpy.ndarray of shape (n,)]:
                containting the index of the cluster in c
                    that each data point belongs to
    """
    return None, None
