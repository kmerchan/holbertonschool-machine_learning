#!/usr/bin/env python3
"""
Defines a function that builds an inception network
using Keras model
"""


import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
    Builds an inception network using Keras model

    input data will have shape (224, 224, 3)
    All convolutions inside and outside should use ReLU activation

    returns:
        the keras model
    """
    init = K.initializers.he_normal()
    activation = K.activations.relu
    img_input = K.Input(shape=(224, 224, 3))
    C0 = K.layers.Conv2D(filters=64,
                         kernel_size=(7, 7),
                         padding='same',
                         strides=(2, 2),
                         activation=activation,
                         kernel_initializer=init)(img_input)
    MP1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(C0)
    C2 = K.layers.Conv2D(filters=64,
                         kernel_size=(1, 1),
                         padding='same',
                         strides=(1, 1),
                         activation=activation,
                         kernel_initializer=init)(MP1)
    C3 = K.layers.Conv2D(filters=192,
                         kernel_size=(3, 3),
                         padding='same',
                         strides=(1, 1),
                         activation=activation,
                         kernel_initializer=init)(C2)
    MP4 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(C3)
    I5 = inception_block(MP4, [64, 96, 128, 16, 32, 32])
    I6 = inception_block(I5, [128, 128, 192, 32, 96, 64])
    MP7 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(I6)
    I8 = inception_block(MP7, [192, 96, 208, 16, 48, 64])
    I9 = inception_block(I8, [160, 112, 224, 24, 64, 64])
    I10 = inception_block(I9, [128, 128, 256, 24, 64, 64])
    I11 = inception_block(I10, [112, 144, 288, 32, 64, 64])
    I12 = inception_block(I11, [256, 160, 320, 32, 128, 128])
    MP13 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                 strides=(2, 2),
                                 padding='same')(I12)
    I14 = inception_block(MP13, [256, 160, 320, 32, 128, 128])
    I15 = inception_block(I14, [384, 192, 384, 48, 128, 128])
    AP16 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                     strides=(1, 1),
                                     padding='same')(I15)
    Dropout17 = K.layers.Dropout(rate=0.4)(AP16)
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(Dropout17)
    model = K.Model(inputs=img_input, outputs=output)
    return model
