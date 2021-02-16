#!/usr/bin/env python3
"""
Defines a function that saves a model's configuration in JSON format
and defines a function that loads a model with specific configuration
using Keras library
"""


import tensorflow.keras as K


def save_config(network, filename):
    """
    Saves a model's configuration in JSON format

    parameters:
        network [keras model]: model to save configuration of
        filename [str]:
            file name where the configuration should be saved
                in JSON format

    returns:
        None
    """
    json = network.to_json()
    with open(filename, 'w+') as f:
        f.write(json)
    return None


def load_config(filename):
    """
    Loads model's weights

    parameters:
        filename [str]:
            path of file containing model's configuration
                in JSON format

    returns:
        the loaded model
    """
    with open(filename, 'r') as f:
        json_string = f.read()
    model = K.models.model_from_json(json_string)
    return model
