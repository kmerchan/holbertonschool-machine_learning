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
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    n = matrix.shape[0]
    positive = 0
    negative = 0
    for i in range(n):
        d_i = np.linalg.det(matrix[:(i + 1), :(i + 1)])
        print("d_i: ", d_i)
        if d_i > 0:
            positive += 1
        if d_i < 0:
            negative += 1
    if positive == n:
        return ("Positive definite")
    if negative == n:
        return ("Negative definite")
    if positive and negative and d_i != 0:
        return ("Indefinite")
    if not negative and d_i == 0:
        return ("Positive semi-definite")
    if not positive and d_i == 0:
        return ("Negative semi-definite")
    return None
