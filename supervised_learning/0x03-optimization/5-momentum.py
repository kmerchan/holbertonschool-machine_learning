#!/usr/bin/env python3
"""
Defines function that updates a variable
using gradient descent with momentum optimization algorithm
"""


def update_variables_momentum(alpha, beta1, var, grad, v):
    """
    Updates a variable using gradient descent
        with momentum optimization algorithm

    parameters:
        alpha [float]: learning rate
        beta1 [float]: momentum weight
        var [numpy.ndarray]:
            contains the variance to be updated
        grad [numpy.ndarray]:
            contains the gradient of var
        v [tf.moment]:
            the previous first moment of var

    returns:
        the updated variable and the new moment, respectively
    """
