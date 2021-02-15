#!/usr/bin/env python3
"""
Defines a function that builds a neural network
using Keras library
"""


import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using Keras library

    parameters:
        nx [int]: number of input features to the network
        layers [list]:
            contains the number of nodes in each layer of the network
        activations [list]:
            contains the activation functions used for each layer
        lambtha [float]:
            the L2 regularization parameter
        keep_prob [float]:
            the probability that a node will be kept for dropout

    returns:
        the keras model
    """
    sequential = []
    shape = (nx,)
    reg_l2 = K.regularizers.l2(lambtha)
    for index, i in enumerate(layers):
        if index is 0:
            sequential.append(K.layers.Dense(layers[i],
                                             activation=activations[i],
                                             kernel_regularizer=reg_l2
                                             input_shape=shape))
        else:
            sequential.append(K.layers.Dense(layers[i],
                                             activation=activations[i],
                                             kernel_regularizer=reg_l2))
    sequential.append(K.layers.Dropout(1 - keep_prob))
    model = K.Sequential(sequential)
    return model
