#!/usr/bin/env python3
"""
Defines function that calculates the minor matrix of a matrix
"""


def minor(matrix):
    """
    Calculates the minor matrix of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose minor matrix should be calculated

    returns:
        the minor matrix of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != height or len(row) is 0:
            raise ValueError("matrix must be a non-empty square matrix")
