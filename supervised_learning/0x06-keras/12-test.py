#!/usr/bin/env python3
"""
Defines a function that tests a neural network
using Keras library
"""


import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """
    Tests a neural network

    parameters:
        network [keras model]: model to test
        data: input data to test the model with
        labels: correct one-hot labels of data
        verbose [boolean]:
            determines if output should be printed during testing process

    returns:
        the loss and accuracy of the model with the testing data
    """
    loss, accuracy = network.evaluate(x=data,
                                      y=labels,
                                      verbose=verbose)
    return loss, accuracy
