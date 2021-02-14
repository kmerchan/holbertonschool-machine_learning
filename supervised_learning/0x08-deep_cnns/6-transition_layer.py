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
    be preceded by batch normalization along the channels axis
    and then ReLU activation, respectively

    All weights should be initialized using he normal

    returns:
        the output of transition layer and
            the number of filter within the output
    """
    init = K.initializers.he_normal()
    activation = K.activations.relu
    Batch_Norm1 = K.layers.BatchNormalization(axis=3)(X)
    ReLU1 = K.layers.Activation(activation)(Batch_Norm1)
    nb_filters *= compression
    nb_filters = int(nb_filters)
    C1 = K.layers.Conv2D(filters=nb_filters,
                         kernel_size=(1, 1),
                         padding='same',
                         kernel_initializer=init)(ReLU1)
    AP1 = K.layers.AveragePooling2D(pool_size=(2, 2),
                                    strides=(2, 2),
                                    padding="valid")(C1)
    return AP1, nb_filters
