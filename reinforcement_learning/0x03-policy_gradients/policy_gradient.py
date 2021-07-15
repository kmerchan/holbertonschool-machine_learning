#!/usr/bin/env python3
"""
Defines function to compute policy with a weight of a matrix
"""


import numpy as np


def policy(matrix, weight):
    """
    Computes policy with a weight of a matrix

    parameters:
        matrix [np.ndarray]: the matrix to compute policy from
        weight [np.ndarray]: the weights to compute policy from

    returns:
        the policy
    """
    # for each column of weights, sum (matrix[i] * weight[i]) using dot product
    dot_product = matrix.dot(weight)
    # find the exponent of the calculated dot product
    exp = np.exp(dot_product)
    # policy is exp / sum(exp)
    policy = exp / np.sum(exp)
    return policy
