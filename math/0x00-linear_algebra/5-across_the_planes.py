#!/usr/bin/env python3
""" defines function that adds two 2D matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ returns new matrix, the sum of two 2D matrices added element-wise """
    mat1_shape = []
    matrix = mat1
    while (type(matrix) is list):
        mat1_shape.append(len(matrix))
        matrix = matrix[0]
    mat2_shape = []
    matrix = mat2
    while (type(matrix) is list):
        mat2_shape.append(len(matrix))
        matrix = matrix[0]

    if mat1_shape != mat2_shape:
        return None
    sum_matrix = []
    for index, dimension in enumerate(mat1_shape):
        for idx, i in enumerate(range(dimension)):
            if idx is 0:
                sum_matrix.append([])
            sum_matrix[index].append(mat1[index][i] + mat2[index][i])
    return sum_matrix
