#!/usr/bin/env python3
"""
Defines function that performs the forward algorithm for a Hidden Markov Model
"""


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """
    Performs the forward algorithm for a Hidden Markov Model

    parameters:
        Observation [numpy.ndarray of shape (T,)]:
            contains the index of the observation
            T: number of observations
        Emission [numpy.ndarray of shape (N, M)]:
            contains the emission probability of a specific observation
                given a hidden state
            N: number of hidden states
            M: number of all possible observations
        Transition [2D numpy.ndarray of shape (N, N)]:
            contains the transition probabilities
            Transition[i, j] is the probabilitiy of transitioning
                from the hidden state i to j
        Initial [numpy.ndarray of shape (N, 1)]:
            contains the probability of starting in a particular hidden state

    returns:
        P, F:
            P [float]:
                the likelihood of the observations given the model
            F [a numpy.ndarray of shape (N. T)]:
                contains the forward path probabilities
                F[i, j] is the probability of being in hidden state i at time j
                    given the previous observations
        or None, None on failure
    """
    # check that Observation is the correct type and dimension
    if type(Observation) is not np.ndarray or len(Observation.shape) < 1:
        return None, None
    # save T from Observation's shape
    T = Observation.shape[0]
    # check that Emission is the correct type and dimension
    if type(Emission) is not np.ndarray or len(Emission.shape) != 2:
        return None, None
    # save N and M from Emission's shape
    N, M = Emission.shape
    # check that Transition is the correct type and dimension
    if type(Transition) is not np.ndarray or len(Transition.shape) != 2:
        return None, None
    # check that Transition's dimensions match N from Emission
    N_check1, N_check2 = Transition.shape
    if N_check1 != N or N_check2 != N:
        return None, None
    # check that Initial is the correct type and dimension
    if type(Initial) is not np.ndarray or len(Initial.shape) != 2:
        return None, None
    # check that Initial's dimensions match (N, 1)
    N_check1, one = Initial.shape
    if N_check1 != N or one != 1:
        return None, None
    return None, None
