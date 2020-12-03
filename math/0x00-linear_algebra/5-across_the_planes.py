#!/usr/bin/env python3
""" defines function that adds two 2D matrices element-wise """


def add_matrices2D(mat1, mat2):
    """ returns new matrix, the sum of two 2D matrices added element-wise """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None
    sum_matrix = []
    for index, dimension in enumerate(matrix_shape(mat1)):
        for idx, i in enumerate(range(dimension)):
            if idx is 0:
                sum_matrix.append([])
            sum_matrix[index].append(mat1[index][i] + mat2[index][i])
    return sum_matrix


def matrix_shape(matrix):
    """ returns list of integers representing dimensions of given matrix """
    matrix_shape = []
    while (type(matrix) is list):
        matrix_shape.append(len(matrix))
        matrix = matrix[0]
    return matrix_shape
