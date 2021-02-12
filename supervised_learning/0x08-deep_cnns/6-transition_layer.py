#!/usr/bin/env python3
"""
Defines a function that builds a transition layer
using Keras
"""


import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    Builds a transition layer using Keras

    parameters:
        X: output from the previous layer
        nb_filters [int]:
            represents the number of filters in X
        compression:
            the compression factor for the transition layer

    Use compression as used for DenseNet-C

    All convolutions inside the dense block should
    be followed by batch normalization along the channels axis
    and then ReLU activation, respectively

    All weights should be initialized using he normal

    returns:
        the output of transition layer and
            the number of filter within the output
    """
    return X, nb_filters
