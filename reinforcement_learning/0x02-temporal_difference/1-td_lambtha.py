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
    # episode = [[states], [rewards]]
    episode = [[], []]
    # set up eligibility traces as a list initialized to 0
    Et = [0 for i in range(env.observation_space.n)]
    for ep in range(episodes):
        # the initial state comes from resetting the environment
        state = env.reset()
        # iterate until done or max number of steps per episode reached
        for step in range(max_steps):
            # list of eligibility traces calculated with lambda & gamma
            Et = list(np.array(Et) * lambtha * gamma)
            # update list by increasing Et at current state
            Et[state] += 1

            # get action from the current state using policy
            action = policy(state)
            # perform the action to get next_state, reward, done, and info
            next_state, reward, done, info = env.step(action)

            # if the algorithm finds a hole, the reward is updated to -1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'H':
                reward = -1
            # if the algorithm finds the goal, the reward is updated to 1
            if env.desc.reshape(env.observation_space.n)[next_state] == b'G':
                reward = 1

            # delta = reward + (gamma * V[next_state]) - V[state]
            delta_t = reward + gamma * V[next_state] - V[state]
            # V[state] = V[state] + (alpha * delta * eligibility trace[state])
            V[state] = V[state] + alpha * delta_t * Et[state]

            # break if done to trigger return
            if done:
                break
            # otherwise, update state to next_state and continue
            state = next_state
    # return V as numpy array when finished
    return np.array(V)
