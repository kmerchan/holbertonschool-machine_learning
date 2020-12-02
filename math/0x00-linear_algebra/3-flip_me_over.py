#!/usr/bin/env python3
""" defines function that transposes a 2D matrix """


def matrix_transpose(matrix):
    """ returns new matrix that is a transpose of the given 2D matrix """
    matrix_transpose = []
    for index, row in enumerate(matrix):
        if index is 0:
            for i in row:
                matrix_transpose.append([])
        for idx, i in enumerate(row):
            matrix_transpose[idx].append(i)
    return matrix_transpose
