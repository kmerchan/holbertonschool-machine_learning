#!/usr/bin/env python3
"""
Defines function that calculates the cofactor matrix of a matrix
"""


def cofactor(matrix):
    """
    Calculates the cofactor matrix of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose cofactor matrix should be calculated

    returns:
        the cofactor matrix of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != height or len(row) is 0:
            raise ValueError("matrix must be a non-empty square matrix")
