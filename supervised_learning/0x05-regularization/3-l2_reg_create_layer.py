#!/usr/bin/env python3
"""
Defines a function that creates a TensorFlow layer
that includes L2 Regularization
"""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    Creates a TensorFlow layer that includes L2 regularization

    parameters:
        prev [tensor]: contains the output of the previous layer
        n [int]: number of nodes the new layer should contain
        activation [activation function]:
            activation function new layer should use
        lambtha: L2 regularization parameter

    returns:
        output of the new layer
    """
    l2_reg = tf.contrib.layers.l2_regularizer(lambtha)
    weights_initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG")
    layer = tf.layers.Dense(
        n,
        activation=activation,
        name="layer",
        kernel_initializer=weights_initializer,
        kernel_regularizer=l2_reg)
    return (layer(prev))
