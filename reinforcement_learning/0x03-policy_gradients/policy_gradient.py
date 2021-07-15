#!/usr/bin/env python3
"""
Defines function to compute Monte Carlo policy gradient based on
state and weight matrices
"""


import numpy as np


def policy(matrix, weight):
    """
    Computes policy with a weight of a matrix

    parameters:
        matrix [numpy.ndarray]: the matrix to compute policy from
        weight [numpy.ndarray]: the weights applied to the matrix

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


def policy_gradient(state, weight):
    """
    Computes the Monte Carlo policy gradient based on the policy
        calculated from the above policy() function

    parameters:
        state [numpy.ndarray]:
            matrix representing the current observation of the environment
        weight [numpy.ndarray]:
            matrix of random weight

    returns:
        the action and the gradient
    """
    # first calculate policy using the policy function above
    policy = policy(state, weight)
    return None, None
