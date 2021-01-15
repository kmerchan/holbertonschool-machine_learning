#!/usr/bin/env python3
"""
Defines function that updates a variable in place
using the Adam optimization algorithm
"""


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """
    Updates a variable in place using Adam optimization algorithm

    parameters:
        alpha [float]: learning rate
        beta1 [float]: weight for first moment
        beta2 [float]: weight for second moment
        epsilon [float]: small number to avoid division by zero
        var [numpy.ndarray]:
            contains the variance to be updated
        grad [numpy.ndarray]:
            contains the gradient of var
        v [tf.moment]:
            the previous first moment of var
        s [tf.moment]:
            the previous second moment of var
        t [int]: time step used for bias correction

    v_dW = (beta * v_dW) + ((1 - beta) * dW)
    s_dW = (beta * s_dW) + ((1 - beta) * (dW ** 2))

    returns:
        the updated variable, the new first moment, and the new second moment,
            respectively
    """
    v_dW = (beta1 * v) + ((1 - beta1) * grad)
    s_dW = (beta2 * s) + ((1 - beta2) * (grad ** 2))
    v_dW_c = v_dW / (1 - (beta1 ** t))
    s_dW_c = s_dW / (1 - (beta2 ** t))
    var -= alpha * (v_dW_c / (epsilon + (s_dW_c ** (1 / 2))))
    return var, v_dW, s_dW
