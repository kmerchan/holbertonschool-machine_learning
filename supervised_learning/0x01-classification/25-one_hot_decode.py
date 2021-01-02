#!/usr/bin/env python3
"""
defines function that converts a one-hot matrix
into a vector of labels
"""


import numpy as np


def one_hot_decode(one_hot):
    """
    converts a one-hot matrix into a numeric vector of labels

    parameters:
        one-hot [numpy.ndarray with shape (classes, m)]:
            one-hot encoded matrix to decode
            classes: the maximum number of classes
            m: the number of examples
    returns:
        numpy.ndarray with shape (m,) containing the numeric labels,
            or None if fails
    """
    if type(one_hot) is not np.ndarray or len(one_hot.shape) != 2:
        return None
    vector = one_hot.transpose().argmax(axis=1)
    return vector
