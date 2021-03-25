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
