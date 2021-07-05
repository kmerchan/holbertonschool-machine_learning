#!/usr/bin/env python3

import gym
import numpy as np
sarsa_lambtha = __import__('2-sarsa_lambtha').sarsa_lambtha

np.random.seed(0)
env = gym.make('FrozenLake8x8-v0')
Q = np.random.uniform(size=(64, 4))
np.set_printoptions(precision=4)
print(sarsa_lambtha(env, Q, 0.9))
