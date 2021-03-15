#!/usr/bin/env python3
"""
Defines function that calculates the inverse of a matrix
"""


def inverse(matrix):
    """
    Calculates the inverse of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose inverse should be calculated

    returns:
        the inverse of matrix or None if matrix is singular
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != height or len(row) is 0:
            raise ValueError("matrix must be a non-empty square matrix")
    return None
