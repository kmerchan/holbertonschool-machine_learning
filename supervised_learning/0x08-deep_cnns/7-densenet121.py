#!/usr/bin/env python3
"""
Defines a function that builds a DenseNet-121 network
using Keras model
"""


import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    Builds a DenseNet-121 network using Keras model

    parameters:
        growth_rate: growth_rate using for DenseNet-B blocks
        compression: compression factor used for transition layers

    input data will have shape (224, 224, 3)

    All convolutions inside and outside blocks should be preceded by
    batch normalization along the channels axis and
    rectified ReLU activation, respectively

    All weights should be initialized with he normal

    returns:
        the keras model
    """
    init = K.initializers.he_normal()
    activation = K.activations.relu
    img_input = K.Input(shape=(224, 224, 3))
    Batch_NormC0 = K.layers.BatchNormalization(axis=3)(img_input)
    ReLUC0 = K.layers.Activation(activation)(Batch_NormC0)
    C0 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         padding='same',
                         strides=(2, 2),
                         kernel_initializer=init)(ReLUC0)
    MP1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(C0)
    DB2, nb_filters = dense_block(MP1, 64, growth_rate, 6)
    TL3, nb_filters = transition_layer(DB2, nb_filters, compression)
    DB4, nb_filters = dense_block(TL3, nb_filters, growth_rate, 12)
    TL5, nb_filters = transition_layer(DB4, nb_filters, compression)
    DB6, nb_filters = dense_block(TL5, nb_filters, growth_rate, 24)
    TL7, nb_filters = transition_layer(DB6, nb_filters, compression)
    DB8, nb_filters = dense_block(TL7, nb_filters, growth_rate, 16)

    AP9 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                    strides=(1, 1),
                                    padding='valid')(DB8)
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(AP9)

    model = K.Model(inputs=img_input, outputs=output)
    return model
