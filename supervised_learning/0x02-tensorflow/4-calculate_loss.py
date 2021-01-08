#!/usr/bin/env python3
"""
Defines a function that calculates the softmax
cross-entropy loss of a prediction
"""


import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    Calculates the softmax cross-entropy loss of a prediction

    parameters:
        y [tf.placeholder]: placeholder for labels of the input data
        y_pred [tensor]: contains network's predictions

    returns:
        tensor containing loss of the prediction
    """
    loss = tf.losses.softmax_cross_entropy(
        y,
        logits=y_pred,
    )
    return loss
