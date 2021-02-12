#!/usr/bin/env python3
"""
Defines a function that builds an identity block
using Keras
"""


import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    Builds an identity block using Keras

    parameters:
        A_prev: output from the previous layer
        filters [tuple or list]:
            containing F11, F3, F12, respectively

    F11: number of filters in the first 1x1 convolution
    F3: number of filters in the 3x3 convolution
    F12: number of filters in the second 1x1 convolution

    All convolutions inside the identity block should
    be followed by batch normalization along the channels axis
    and then ReLU activation, respectively

    All weights should be initialized using he normal

    returns:
        the activated output of the identity block
    """
    F11, F3, F12 = filters
    init = K.initializers.he_normal()
    activation = K.activations.relu
    C11 = K.layers.Conv2D(filters=F11,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=init)(A_prev)
    Batch_Norm11 = K.layers.BatchNormalization(axis=3)(C11)
    ReLU11 = K.layers.Activation(activation)(Batch_Norm11)
    C3 = K.layers.Conv2D(filters=F3,
                         kernel_size=(3, 3),
                         padding='same',
                         kernel_initializer=init)(ReLU11)
    Batch_Norm3 = K.layers.BatchNormalization(axis=3)(C3)
    ReLU3 = K.layers.Activation(activation)(Batch_Norm3)
    C12 = K.layers.Conv2D(filters=F12,
                          kernel_size=(1, 1),
                          padding='same',
                          kernel_initializer=init)(ReLU3)
    Batch_Norm12 = K.layers.BatchNormalization(axis=3)(C12)
    Addition = K.layers.Add()([Batch_Norm12, A_prev])
    output = K.layers.Activation(activation)(Addition)
    return output
