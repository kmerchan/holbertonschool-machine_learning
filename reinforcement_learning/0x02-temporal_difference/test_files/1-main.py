#!/usr/bin/env python3

import gym
import numpy as np
td_lambtha = __import__('1-td_lambtha').td_lambtha

np.random.seed(0)

env = gym.make('FrozenLake8x8-v0')
LEFT, DOWN, RIGHT, UP = 0, 1, 2, 3

def policy(s):
    p = np.random.uniform()
    if p > 0.5:
        if s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s // 8 != 0 and env.desc[s // 8 - 1, s % 8] != b'H':
            return UP
        else:
            return LEFT
    else:
        if s // 8 != 7 and env.desc[s // 8 + 1, s % 8] != b'H':
            return DOWN
        elif s % 8 != 7 and env.desc[s // 8, s % 8 + 1] != b'H':
            return RIGHT
        elif s % 8 != 0 and env.desc[s // 8, s % 8 - 1] != b'H':
            return LEFT
        else:
            return UP

V = np.where(env.desc == b'H', -1, 1).reshape(64).astype('float64')
np.set_printoptions(precision=4)
print(td_lambtha(env, V, policy, 0.9).reshape((8, 8)))
