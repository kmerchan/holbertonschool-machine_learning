#!/usr/bin/env python3
"""
Defines a function that calculates the likelihood of obtaining this data
given various hypothetical probabilities of developing severe side effects
"""


import numpy as np


def likelihood(x, n, P):
    """
    Calculates the likelihood of obtaining this data given
    various hypothetical probabilities of developing severe side effects

    parameters:
        x [int]: total number of patients that develop severe side effects
        n [int]: total number of patients observed
        P [1D numpy.ndarray]: containing the various hypothetical probabilities
            of developing severe side effects

    returns:
        a 1D numpy.ndarray containing the likelihood of obtaining the data,
            x and n, for each probability in P
    """
    if type(n) is not int or n <= 0:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    for value in P:
        if value > 1 or value < 0:
            raise ValueError("All values in P must be in the range [0, 1]")
    # likelihood calculated as binomial distribution
    factorial = np.math.factorial
    fact_coefficient = factorial(n) / (factorial(n - x) * factorial(x))
    likelihood = fact_coefficient * (P ** x) * ((1 - P) ** (n - x))
    return likelihood
