#!/usr/bin/env python3
""" defines a function that calculates a summation """


def summation_i_squared(n):
    """ calculates summation of i^2 from i=1 to n """
    if type(n) is not int:
        return None
    return sigma_recursion(n, 0)


def sigma_recursion(n, sigma_sum):
    """ calculate summation recursively """
    if n is 1:
        return (sigma_sum + 1)
    sigma_sum += n**2
    return sigma_recursion(n - 1, sigma_sum)
