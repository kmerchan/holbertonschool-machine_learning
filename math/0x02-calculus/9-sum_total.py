#!/usr/bin/env python3
""" defines a function that calculates a summation """


def summation_i_squared(n):
    """ calculates summation of i^2 from i=1 to n """
    if type(n) is not int:
        return None
    if n is 1:
        return (1)
    sigma_sum = pow(n, 2)
    sigma_sum += summation_i_squared(n - 1)
    return sigma_sum
