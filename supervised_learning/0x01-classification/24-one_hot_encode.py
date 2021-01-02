#!/usr/bin/env python3
"""
defines function that converts a numeric label vector
into a one-hot matrix
"""


import numpy as np


def one_hot_encode(Y, classes):
    """
    converts a numeric label vector into a one-hot matrix

    parameters:
        Y [numpy.ndarray with shape (m,)]: contains numeric class labels
            m is the number of examples
        classes [int]: the maximum number of classes found in Y

    returns:
        one-hot encoding of Y with shape (classes, m)
            or None if fails
    """
    # if type(Y) is not np.ndarray or len(Y.shape) != 1 or len(Y) < 1:
    # return None
    # if type(classes) is not int or classes != (Y.max() + 1):
    # return None
    # one_hot = np.eye(classes)[Y].transpose()
    # return one_hot
    if type(Y) is not np.ndarray:
        return None
    if type(classes) is not int:
        return None
    try:
        one_hot = np.eye(classes)[Y].transpose()
        return one_hot
    except Exception as err:
        return None
