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

    v_dW = (beta * v_dW) + ((1 - beta) * dW)
    W = W - (alpha * v_dW)

    returns:
        the updated variable and the new moment, respectively
    """
    dW_prev = (beta1 * v) + ((1 - beta1) * grad)
    var -= (alpha * dW_prev)
    return var, dW_prev
