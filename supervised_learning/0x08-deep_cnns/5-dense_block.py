#!/usr/bin/env python3
"""
Defines a function that builds a dense block
using Keras
"""


import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    Builds a dense block using Keras

    parameters:
        X: output from the previous layer
        nb_filters [int]:
            represents the number of filters in X
        growth_rate: growth rate for the dense block
        layers: number of layers in dense block

    Use the bottleneck layers used for DenseNet-B

    All convolutions inside the dense block should
    be followed by batch normalization along the channels axis
    and then ReLU activation, respectively

    All weights should be initialized using he normal

    returns:
        the concatenated output of each layer within the dense block
            and the number of filter within the concatenated outputs
    """
    return X, nb_filters
