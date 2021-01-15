#!/usr/bin/env python3
"""
Defines function that updates a variable
using RMSProp optimization algorithm
"""


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """
    Updates a variable using RMSProp optimization algorithm

    parameters:
        alpha [float]: learning rate
        beta2 [float]: RMSProp weight
        epsilon [float]:
            small number to avoid division by zero
        var [numpy.ndarray]:
            contains the variance to be updated
        grad [numpy.ndarray]:
            contains the gradient of var
        s [tf.moment]:
            the previous second moment of var

    s_dW = (beta * s_dW) + ((1 - beta) * (dW ** 2))
    W = W - (alpha * (dW / sqrt(s_dW)))

    returns:
        the updated variable and the new moment, respectively
    """
    s_dW = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    var -= alpha * (grad / (epsilon + (s_dW ** (1 / 2))))
    return var, s_dW
