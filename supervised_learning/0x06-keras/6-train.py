#!/usr/bin/env python3
"""
Updates function that trains a model using mini-batch gradient descent
to train using early stopping with Keras library
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent,
        including analyzing validation data and using early stopping

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
        early_stopping [boolean]:
            indicated whether early stopping should be used
            early stopping should only be performed if validation_data exists
            early stopping should be based on validation loss
        patience:
            patience used for early stopping
        verbose [boolean]:
            determines if output should be printed during training
        shuffle [boolean]:
            determines whether to shuffle the batches every epoch

    returns:
        the History object generated after training the model
    """
    if early_stopping and validation_data:
        callback = []
        callback.append(
            K.callbacks.EarlyStopping(monitor='loss', patience=patience))
    else:
        callback = None

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
