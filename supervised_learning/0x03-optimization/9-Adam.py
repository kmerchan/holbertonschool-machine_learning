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

    returns:
        the updated variable, the new first moment, and the new second moment,
            respectively
    """
