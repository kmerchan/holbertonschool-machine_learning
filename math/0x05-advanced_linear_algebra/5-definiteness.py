#!/usr/bin/env python3
"""
Defines function that calculates the definiteness of a matrix
"""


import numpy as np


def definiteness(matrix):
    """
    Calculates the definiteness of a matrix

    parameters:
        matrix [numpy.ndarray of shape(n, n)]:
            matrix whose definiteness should be calculated

    returns:
        one of the following strings indicating definiteness or None:
            "Positive definite"
            "Positive semi-definite"
            "Negative definite"
            "Negative semi-definite"
            "Indefinite"
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    return None
