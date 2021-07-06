#!/usr/bin/env python3
"""
Defines function to perform the SARSA(λ) algorithm
"""


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100, alpha=0.1,
                  gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm

    parameters:
        env: the openAI environment instance
        Q [numpy.ndarray of shape(s, a)]: contains the Q table
        lambtha: the eligibility trace factor
        episodes [int]: total number of episodes to train over
        max_steps [int]: the maximum number of steps per episode
        alpha [float]: the learning rate
        gamma [float]: the discount rate
        epsilon: the initial threshold for epsilon greedy
        min_epsilon [float]: the minimum value that epsilon should decay to
        epsilon_decay [float]: decay rate for updating epsilon between episodes

    returns:
        Q: the updated Q table
    """
    return None
