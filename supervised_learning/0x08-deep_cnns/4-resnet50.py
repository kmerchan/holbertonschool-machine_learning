#!/usr/bin/env python3
"""
Defines a function that builds a ResNet-50 network
using Keras model
"""


import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Builds a ResNet-50 network using Keras model

    input data will have shape (224, 224, 3)

    All convolutions inside and outside blocks should be followed by
    batch normalization along the channels axis and
    rectified ReLU activation, respectively

    All weights should be initialized with he normal

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
                         kernel_initializer=init)(img_input)
    Batch_NormC0 = K.layers.BatchNormalization(axis=3)(C0)
    ReLUC0 = K.layers.Activation(activation)(Batch_NormC0)
    MP1 = K.layers.MaxPooling2D(pool_size=(3, 3),
                                strides=(2, 2),
                                padding='same')(ReLUC0)
    PB2 = projection_block(MP1, [64, 64, 256], s=1)
    IB3 = identity_block(PB2, [64, 64, 256])
    IB4 = identity_block(IB3, [64, 64, 256])

    PB5 = projection_block(IB4, [128, 128, 512], s=2)
    IB6 = identity_block(PB5, [128, 128, 512])
    IB7 = identity_block(IB6, [128, 128, 512])
    IB8 = identity_block(IB7, [128, 128, 512])

    PB9 = projection_block(IB8, [256, 256, 1024], s=2)
    IB10 = identity_block(PB9, [256, 256, 1024])
    IB11 = identity_block(IB10, [256, 256, 1024])
    IB12 = identity_block(IB11, [256, 256, 1024])
    IB13 = identity_block(IB12, [256, 256, 1024])
    IB14 = identity_block(IB13, [256, 256, 1024])

    PB15 = projection_block(IB14, [512, 512, 2048], s=2)
    IB16 = identity_block(PB15, [512, 512, 2048])
    IB17 = identity_block(IB16, [512, 512, 2048])

    AP18 = K.layers.AveragePooling2D(pool_size=(7, 7),
                                     strides=(1, 1),
                                     padding='valid')(IB17)
    output = K.layers.Dense(1000,
                            activation='softmax',
                            kernel_initializer=init)(AP18)
    model = K.Model(inputs=img_input, outputs=output)
    return model
