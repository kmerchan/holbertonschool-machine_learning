#!/usr/bin/env python3
"""
Defines a function to return two placeholders for the neural network
"""


import tensorflow as tf


def create_placeholders(nx, classes):
    """
    Returns two placeholders, x and y, for the neural network
    x is the placeholder for input data to the neural network
    y is the placeholder for the one-hot labels for the input data

    parameters:
        nx [int]: the number of feature columns in the data
        classes [int]: the number of classes in the classifier

    returns:
        the placeholders, x and y, respectively
    """
    x = tf.placeholder("float", shape=(None, nx), name="x")
    y = tf.placeholder("float", shape=(None, classes), name="y")
    return x, y
