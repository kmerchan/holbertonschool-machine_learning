#!/usr/bin/env python3
"""
Defines function that performs the Baum-Welch algorithm for Hidden Markov Model
"""


import numpy as np


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """
    Performs the Baum-Welch algorithm for a Hidden Markov Model

    parameters:
        Observation [numpy.ndarray of shape (T,)]:
            contains the index of the observation
            T: number of observations
        Transition [2D numpy.ndarray of shape (M, M)]:
            contains the initialized transition probabilities
            M: the number of hidden states
        Emission [numpy.ndarray of shape (M, N)]:
            contains the initialized emission probabilities
            N: number of output states
        Initial [numpy.ndarray of shape (M, 1)]:
            contains the initialized starting probabilities
        iterations [positive int]:
            the number of times expectation-maximization should be performed

    returns:
        the converged Transition, Emission
        or None, None on failure
    """
    # check that Observation is the correct type and dimension
    if type(Observation) is not np.ndarray or len(Observation.shape) < 1:
        return None, None
    # save T from Observation's shape
    T = Observation.shape[0]
    # check that Transition is the correct type and dimension
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    # save M and check that Transition is square
    M, M_check = Transition.shape
    if M != M_check:
        return None, None
    # check that Emission is the correct type and dimension
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    # check that Emission's dimension matches N from Transition and save N
    M_check, N = Emission.shape
    if M_check != M:
        return None, None
    # check that Initial is the correct type and dimension
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    # check that Initial's dimensions match (M, 1)
    M_check, one = Initial.shape
    if M_check != M or one != 1:
        return None, None
    # check that iterations is a positive int
    if type(iterations) is not int or iterations < 1:
        return None, None
    return None, None
