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
        beta1 [float]: momentum weight
        epsilon [float]:
            small number to avoid division by zero
        var [numpy.ndarray]:
            contains the variance to be updated
        grad [numpy.ndarray]:
            contains the gradient of var
        s [tf.moment]:
            the previous second moment of var

    returns:
        the updated variable and the new moment, respectively
    """
