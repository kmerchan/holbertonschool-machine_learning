#!/usr/bin/env python3
"""
Defines a function that builds an inception block
using Keras
"""


import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    Builds an inception block using Keras

    parameters:
        A_prev: output from the previous layer
        filters [tuple or list]:
            containing F1, F3R, F3, F5R, F5, FPP, respectively

    F1: number of filters in the 1x1 convolution
    F3R: number of filters in the 1x1 convolution before the 3x3 convolution
    F3: number of filters in the 3x3 convolution
    F5R: number of filters in the 1x1 convoltion before the 5x5 convolution
    F5: number of filters in the 5x5 convolution
    FPP: number of filters in the 1x1 convoltion after max pooling

    All convolutions inside the inception block should use ReLU activation

    returns:
        the conatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters
    layer_1 = K.layers.Conv2D(filters=F1,
                              kernel_size=(1, 1),
                              padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.he_normal())
    output_1 = layer_1(A_prev)
    layer_2 = K.layers.Conv2D(filters=F3R,
                              kernel_size=(1, 1),
                              padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.he_normal())
    output_2 = layer_2(A_prev)
    layer_3 = K.layers.Conv2D(filters=F3,
                              kernel_size=(3, 3),
                              padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.he_normal())
    output_3 = layer_3(output_2)
    layer_4 = K.layers.Conv2D(filters=F5R,
                              kernel_size=(1, 1),
                              padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.he_normal())
    output_4 = layer_4(A_prev)
    layer_5 = K.layers.Conv2D(filters=F5,
                              kernel_size=(5, 5),
                              padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.he_normal())
    output_5 = layer_5(output_4)
    layer_6 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                    strides=(1, 1),
                                    padding='same')
    output_6 = layer_6(A_prev)
    layer_7 = K.layers.Conv2D(filters=FPP,
                              kernel_size=(1, 1),
                              padding='same',
                              activation=K.activations.relu,
                              kernel_initializer=K.initializers.he_normal())
    output_7 = layer_7(output_6)
    return (K.layers.concatenate([output_1, output_3, output_5, output_7]))
