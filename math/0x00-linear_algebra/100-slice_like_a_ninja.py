#!/usr/bin/env python3
""" defines function that slices a matrix along specific axes using numpy """


def np_slice(matrix, axes={}):
    """ returns numpy.ndarray, the slice of a matrix along specific axes """
    dimensions = len(matrix.shape)
    slices_matrix = dimensions * [slice(None)]
    for axis, value in axes.items():
        slices_matrix[axis] = slice(*value)
    return matrix[tuple(slices_matrix)]
