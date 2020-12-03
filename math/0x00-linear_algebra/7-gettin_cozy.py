#!/usr/bin/env python3
""" defines function that concatenates two 2D matrices along an axis """


def cat_matrices2D(mat1, mat2, axis=0):
    """ returns new matrix that is the concatenation of two 2D matrices """
    if axis is 0:
        # concatenate rows
        for row in mat1:
            mat1_columns = len(row)
        for row in mat2:
            mat2_columns = len(row)
        if mat1_columns != mat2_columns:
            return None
        cat_matrix = []
        for index1, row in enumerate(mat1):
            cat_matrix.append([])
            for i in row:
                cat_matrix[index1].append(i)
        index1 += 1
        for index2, row in enumerate(mat2):
            cat_matrix.append([])
            for i in row:
                cat_matrix[index1 + index2].append(i)
        return cat_matrix
    if axis is 1:
        # concatenates columns
        if len(mat1) != len(mat2):
            return None
        cat_matrix = []
        for index, row in enumerate(mat1):
            cat_matrix.append([])
            for i in mat1[index]:
                cat_matrix[index].append(i)
            for i in mat2[index]:
                cat_matrix[index].append(i)
        return cat_matrix
    return None
