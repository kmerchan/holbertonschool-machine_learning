#!/usr/bin/env python3
""" defines function that adds two 2D matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ returns new matrix, the sum of two 2D matrices added element-wise """
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    sum_matrix = []
    for index, row in enumerate(mat1):
        sum_matrix.append([])
        for i in range(len(row)):
            sum_matrix[index].append(mat1[index][i] + mat2[index][i])
    return sum_matrix
