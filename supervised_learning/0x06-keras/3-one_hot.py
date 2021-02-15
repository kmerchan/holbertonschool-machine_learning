#!/usr/bin/env python3
"""
Defines a function that converts a label vector into a one-hot matrix
using Keras library
"""


import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Converts a label vector into a one-hot matrix using Keras

    parameters:
        labels [vector]:
            contains labels to convert into one-hot matrix
        classes:
            classes for one-hot matrix

    last dimension of the one-hot matrix must be the number of classes

    returns:
        one-hot matrix
    """
    one_hot = K.utils.to_categorical(labels, num_classes=classes)
    return one_hot
