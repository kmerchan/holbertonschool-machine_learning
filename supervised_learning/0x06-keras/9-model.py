#!/usr/bin/env python3
"""
Defines a function that saves an entire model
and defines a function that loads an entire model
using Keras library
"""


import tensorflow.keras as K


def save_model(network, filename):
    """
    Saves the entire model

    parameters:
        network [keras model]: model to save
        filename [str]:
            file name where the model should be saved

    returns:
        None
    """
    network.save(filename)
    return None


def load_model(filename):
    """
    Loads an entire model

    parameters:
        filename [str]:
            file name where the model should be loaded from

    returns:
        the loaded model
    """
    model = K.models.load_model(filename)
    return model
