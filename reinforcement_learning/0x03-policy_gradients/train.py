#!/usr/bin/env python3
"""
Defines function to implement full training with policy gradient
"""


import numpy as np
from policy_gradient import policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implements full training using policy gradient

    parameters:
        env: initial environment
        nb_episodes [int]: the number of episodes used for training
        alpha [float]: the learning rate
        gamma [float]: the discount factor
        show_result [boolean]:
            determines if the environment is rendered every 1000 episodes

    returns:
        all values of the score (sum of all rewards during one episode loop)
    """
    return None
