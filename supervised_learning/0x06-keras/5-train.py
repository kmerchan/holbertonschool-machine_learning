#!/usr/bin/env python3
"""
Updates function that trains a model using mini-batch gradient descent
to also analyze validation data using Keras library
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent,
        including analyzing validation data

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
        validation_data:
            data to be analyzed during model training
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
                          validation_data=validation_data,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
