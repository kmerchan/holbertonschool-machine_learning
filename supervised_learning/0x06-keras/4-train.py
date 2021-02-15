#!/usr/bin/env python3
"""
Defines a function that trains a model using mini-batch gradient descent
using Keras library
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent with Keras

    parameters:
        network [keras model]: model to train
        data [numpy.ndarray of shape (m, nx)]:
            contains the input data
        labels [one-hot numpy.ndarray of shape (m, classes)]:
            contains labels of data
        batch_size [int]:
            size of batch used for mini-batch gradient descent
        epochs [int]:
            number of passes through data for mini-batch gradient descent
        verbose [boolean]:
            determines if output should be printed during training
        shuffle [boolean]:
            determines whether to shuffle the batches every epoch

    returns:
        the History object generated after training the model
    """
    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
