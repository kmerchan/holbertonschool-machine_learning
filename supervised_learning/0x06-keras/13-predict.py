#!/usr/bin/env python3
"""
Defines a function that makes a prediction using neural network
using Keras library
"""


import tensorflow.keras as K


def predict(network, data, verbose=False):
    """
    Makes a prediction using a neural network

    parameters:
        network [keras model]: model to make prediction with
        data: input data to make prediction with
        verbose [boolean]:
            determines if output should be printed during prediction process

    returns:
        the prediction for the data
    """
    prediction = network.predict(x=data,
                                 verbose=verbose)
    return prediction
