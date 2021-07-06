#!/usr/bin/env python3
"""
Defines function to perform the Monte Carlo algorithm
"""


def monte_carlo(env, V, policy, episodes=5000, max_steps=100,
                alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm

    parameters:
        env: the openAI environment instance
        V [numpy.ndarray of shape(s,)]: contains the value estimate
        policy: function that takes in state & returns the next action to take
        episodes [int]: total number of episodes to train over
        max_steps [int]: the maximum number of steps per episode
        alpha [float]: the learning rate
        gamma [float]: the discount rate

    returns:
        V: the updated value estimate
    """
    return None
