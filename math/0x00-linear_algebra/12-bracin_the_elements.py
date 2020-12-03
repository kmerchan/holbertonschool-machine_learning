#!/usr/bin/env python3
""" defines function that performs element-wise operations on two matrices """


def np_elementwise(mat1, mat2):
    """
    performs element-wise operations:
        addition, subtraction, multiplication, and division
    on two matrices, interpreted as numpy.ndarrys

    returns: tuple containing element-wise sum, difference, product, quotient
    """
    result = []
    result.append(mat1 + mat2)
    result.append(mat1 - mat2)
    result.append(mat1 * mat2)
    result.append(mat1 / mat2)
    return tuple(result)
