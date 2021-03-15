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
    if height is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) != height:
            raise ValueError("matrix must be a non-empty square matrix")
    if height is 1:
        return [[1]]
    minor_matrix = []
    for row_i in range(height):
        minor_row = []
        for column_i in range(height):
            sub_matrix = []
            for row in range(height):
                if row == row_i:
                    continue
                new_row = []
                for column in range(height):
                    if column == column_i:
                        continue
                    new_row.append(matrix[row][column])
                sub_matrix.append(new_row)
            minor_row.append(determinant(sub_matrix))
        minor_matrix.append(minor_row)
    return minor_matrix


def determinant(matrix):
    """
    Calculates the determinant of a matrix

    parameters:
        matrix [list of lists]:
            matrix whose determinant should be calculated

    returns:
        the determinant of matrix
    """
    if type(matrix) is not list:
        raise TypeError("matrix must be a list of lists")
    height = len(matrix)
    if height is 0:
        raise TypeError("matrix must be a list of lists")
    for row in matrix:
        if type(row) is not list:
            raise TypeError("matrix must be a list of lists")
        if len(row) is 0 and height is 1:
            return 1
        if len(row) != height:
            raise ValueError("matrix must be a square matrix")
    if height is 1:
        return matrix[0][0]
    if height is 2:
        a = matrix[0][0]
        b = matrix[0][1]
        c = matrix[1][0]
        d = matrix[1][1]
        return ((a * d) - (b * c))
    multiplier = 1
    d = 0
    for i in range(height):
        element = matrix[0][i]
        sub_matrix = []
        for row in range(height):
            if row == 0:
                continue
            new_row = []
            for column in range(height):
                if column == i:
                    continue
                new_row.append(matrix[row][column])
            sub_matrix.append(new_row)
        d += (element * multiplier * determinant(sub_matrix))
        multiplier *= -1
    return (d)
