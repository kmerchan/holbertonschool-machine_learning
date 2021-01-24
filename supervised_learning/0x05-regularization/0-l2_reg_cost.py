#!/usr/bin/env python3
"""
Defines a function that calculates the cost of a neural network
using L2 Regularization
"""

def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    Calculates the cost of a neural network with L2 regularization

    parameters:
        cost: the cost of the network without L2 regularization
        lambtha: the regularization parameter
        weights: a dictionary of the weights and biases of the neural network
        L: the number of layers in the neural network
        m: the number of data points used

    returns:
        the cose of the network accounting for L2 regularization
    """
