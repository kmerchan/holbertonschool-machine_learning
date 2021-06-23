#!/usr/bin/env python3
"""
Defines function that uses epsilon-greedy to determine the next action
"""


import gym
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    Uses epsilon-greedy to determine the next action

    returns:
        the next action index
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        # exploring
        action = np.random.randint(Q.shape[1])
    else:
        # exploiting
        action = np.argmax(Q[state, :])
    return action
