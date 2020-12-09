#!/usr/bin/env python3
""" defines a function that calculates a summation """


def summation_i_squared(n):
    """
    calculates summation of i^2 from i=1 to n

    utilizes Faulhaber's formula for power of 2:
        sum of i^2 from i=1 to n = (n * (n + 1) * (2n + 1)) / 6
                                   or ((n^3) / 3) + ((n^2) / 2) + (n / 6)
    """
    if type(n) is not int or n < 1:
        return None
    sigma_sum = (n * (n + 1) * ((2 * n) + 1)) / 6
    return int(sigma_sum)
