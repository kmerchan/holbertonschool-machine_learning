#!/usr/bin/env python3
"""
Defines function to perform the TD(λ) algorithm
"""


import gym
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """
    Performs the TD(λ) algorithm

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
    episode = [[], []]
    Et = [0 for i in range(env.observation_space.n)]
    for ep in range(episodes):
        state = env.reset()
        for step in range(max_steps):
            Et = list(np.array(Et) * lambtha * gamma)
            Et[state] += 1

            action = policy(state)
            next_state, reward, done, info = env.step(action)

            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1

            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1

            delta_t = reward + gamma * V[next_state] - V[state]

            V[state] = V[state] + alpha * delta_t * Et[state]

            if done:
                break
            state = next_state
    return np.array(V)
