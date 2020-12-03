#!/usr/bin/env python3
""" defines function that performs matrix multiplication """


def mat_mul(mat1, mat2):
    """ returns new matrix that is the product of two 2D matrices """
    mat1_columns = len(mat1[0])
    mat2_rows = len(mat2)
    if mat1_columns != mat2_rows:
        return None
    new_matrix = []
    for row_count, row in enumerate(mat1):
        new_matrix.append([])
        for column_count in range(len(mat2[0])):
            dot = 0
            for index in range(mat1_columns):
                dot += (mat1[row_count][index] * mat2[index][column_count])
            new_matrix[row_count].append(dot)
    return new_matrix
