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
    for layer in layers:
        Batch_Norm1 = K.layers.BatchNormalization(axis=3)(X)
        ReLU1 = K.layers.Activation(activation)(Batch_Norm1)
        C1 = K.layers.Conv2D(filters=(4 * growth_rate),
                             kernel_size=(1, 1),
                             padding='same',
                             kernel_initializer=init)(ReLU1)
        Batch_Norm3 = K.layers.BatchNormalization(axis=3)(C1)
        ReLU3 = K.layers.Activation(activation)(Batch_Norm3)
        C3 = K.layers.Conv2D(filters=growth_rate,
                             kernel_size=(3, 3),
                             padding='same',
                             kernel_initializer=init)(ReLU3)
        X = K.layers.concatenate([X, C3])
        nb_filters += growth_rate
    return X, nb_filters
