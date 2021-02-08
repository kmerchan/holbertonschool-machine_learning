#!/usr/bin/env python3
"""
Defines a function that builds a modified version of LeNet-5 architecture
using Keras
"""


import tensorflow.keras as K


def lenet5(X):
    """
    Builds a modified version of LeNet-5 architecture using Keras

    parameters:
        X [K.input of shape (m, 28, 28, 1)]:
            contains the input images for the network
            m: number of images

    model layers:
    C1: convolutional layer with 6 kernels of shape (5, 5) with same padding
    P2: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    C3: convolutional layer with 16 kernels of shape (5, 5) with valid padding
    P4: max pooling layer with kernels of shape (2, 2) with (2, 2) strides
    F5: fully connected layer with 120 nodes
    F6: fully connected layer with 84 nodes
    F7: fully connected softmax output layer with 10 nodes

    All layers requiring init should initialize kernels with he_normal method
    All hidden layer requiring activation should use relu activation function

    returns:
        K.Model compiled to use Adam optimization (default hyperparameters)
            and accuracy metrics
    """
    weights_initializer = K.initializers.he_normal()
    C1 = K.layers.Conv2D(filters=6,
                         kernel_size=(5, 5),
                         padding='same',
                         activation=K.activations.relu,
                         kernel_initializer=weights_initializer)
    output_1 = C1(X)
    P2 = K.layers.MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2))
    output_2 = P2(output_1)
    C3 = K.layers.Conv2D(filters=16,
                         kernel_size=(5, 5),
                         padding='valid',
                         activation=K.activations.relu,
                         kernel_initializer=weights_initializer)
    output_3 = C3(output_2)
    P4 = K.layers.MaxPooling2D(pool_size=(2, 2),
                               strides=(2, 2))
    output_4 = P4(output_3)
    output_42 = K.layers.Flatten()(output_4)
    F5 = K.layers.Dense(
        120,
        activation=K.activations.relu,
        kernel_initializer=weights_initializer)
    output_5 = F5(output_42)
    F6 = K.layers.Dense(
        84,
        activation=K.activations.relu,
        kernel_initializer=weights_initializer)
    output_6 = F6(output_5)
    F7 = K.layers.Dense(
        10,
        kernel_initializer=weights_initializer)
    output_7 = F7(output_6)
    softmax = K.layers.Softmax()(output_7)
    model = K.Model(inputs=X, outputs=softmax)
    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
