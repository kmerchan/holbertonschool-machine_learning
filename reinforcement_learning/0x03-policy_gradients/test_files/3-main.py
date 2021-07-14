#!/usr/bin/env python3
"""
Main file
"""
import gym

from train import train

env = gym.make('CartPole-v1')

scores = train(env, 10000, 0.000045, 0.98, True)

env.close()
