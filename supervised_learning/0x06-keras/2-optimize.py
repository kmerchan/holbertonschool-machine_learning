#!/usr/bin/env python3
"""
Defines a function that sets up Adam optimizer for Keras model
with categorical crossentropy loss and accuracy metrics
"""


import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    Sets up Adam optimizer for Keras model with
        categorical crossentropy loss and accuracy metrics

    parameters:
        network [keras model]: model to optimize
        alpha [float]: learning rate
        beta1: first Adam optimization parameter
        beta2: second Adam optimization parameter

    returns:
        None
    """
    network.compile(optimizer=K.optimizers.Adam(lr=alpha,
                                                beta_1=beta1,
                                                beta_2=beta2),
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
