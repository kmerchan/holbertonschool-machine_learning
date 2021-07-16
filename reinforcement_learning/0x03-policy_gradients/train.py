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
    # assign weight randomly
    weight = np.random.rand(4, 2)
    # create all_scores to track all values of episode scores
    all_scores = []
    # iterate over all episodes
    for episode in range(nb_episodes):
        # set initial state from environment reset
        state = env.reset()[None, :]
        # set gradients, rewards, and sum of rewards (score) to empty/zero
        gradients = []
        rewards = []
        sum_rewards = 0
        while True:
            # if show_result, render the environment every 1000 episodes
            if show_result and (episode % 1000 == 0):
                env.render()
            # get action and gradient from policy_gradient function
            action, gradient = policy_gradient(state, weight)
            # use action to determine next state, reward, done, and info
            next_state, reward, done, info = env.step(action)
            # append gradient and reward to repective lists
            gradients.append(gradient)
            rewards.append(reward)
            # add the reward to the episode score
            sum_rewards += reward
            # if done, breaks the loop
            if done:
                break
            # else the state is reset to the next state
            state = next_state[None, :]
        # calculate the new weights with gradients and rewards from episode
        for i in range(len(gradients)):
            weight += (alpha * gradients[i] *
                       sum([r * (gamma ** r) for t, r in enumerate(
                           rewards[i:])]))
        # append the episode's score to all_scores
        all_scores.append(sum_rewards)
        # print the current episode and episode's score
        print("{}: {}".format(episode, sum_rewards), end="\r", flush=False)
    # return all values of scores
    return all_scores
