#!/usr/bin/env python3
"""
Defines a function that saves a model's weights
and defines a function that loads a model's weightsl
using Keras library
"""


import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """
    Saves the model's weights

    parameters:
        network [keras model]: model to save weights from
        filename [str]:
            file name where the weights should be saved
        save_format [str]:
            format in which the weights should be saved

    returns:
        None
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """
    Loads model's weights

    parameters:
        network [keras model]: model to load weights to
        filename [str]:
            file name where the weights should be loaded from

    returns:
        None
    """
    network.load_weights(filename)
    return None
