#!/usr/bin/env python3
"""
Defines function that creates the training op
for a neural network in TensorFlow using
the Adam optimization algorithm
"""


import tensorflow as tf


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """
    Creates the training operation for a neural network in TensorFlow
        using the Adam optimization algorithm

    parameters:
        loss: the loss of the network
        alpha [float]: learning rate
        beta1 [float]: weight used for first moment
        beta2 [float]: weight used for second moment
        epsilon [float]: small number to avoid division by zero

    returns:
        the Adam optimization operation
    """
    op = tf.train.AdamOptimizer(
        alpha, beta1=beta1, beta2=beta2, epsilon=epsilon).minimize(loss)
    return op
