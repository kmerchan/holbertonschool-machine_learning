#!/usr/bin/env python3
"""
Defines function that creates a batch normalization layer
for a neural network in TensorFlow
"""


import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow

    parameters:
        prev [tf.moment]: the activated output of the previous layer
        n [int]: the number of nodes in the layer to be created
        activation:
            activation function that should be used on the output of the layer

    utilize tf.layers.Dense layer as the base layer with
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    incorporate two trainable parameters:
        gamma: initialized as vectors of 1
        beta: initialized as vectors of 0
    epsilon = 1 x 10^-8

    returns:
        a tensor of the activated output for the layer
    """
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernal_initializer=weights_initializer)
    x = layer[prev]
    gamma = tf.Variable(tf.constant(
        1, shape=(1, n), trainable=True, name="gamma"))
    beta = tf.Variable(tf.constant(
        0, shape=(1, n), trainable=True, name="gamma"))
    Z = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-8)
    return Z
