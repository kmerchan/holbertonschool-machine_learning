#!/usr/bin/env python3
"""
Defines function that builds, trains, and saves a neural network model
using TensorFlow using Adam optimization, mini-batch gradient descent,
learning rate decay, and batch normalization
"""


import tensorflow as tf
import numpy as np


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in TensorFlow using
        Adam optimization,
        mini-batch gradient descent,
        learning rate decay, and
        batch normalization

    parameters:

    returns:
        the path where the model was saved
    """
