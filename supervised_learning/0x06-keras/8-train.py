#!/usr/bin/env python3
"""
Updates function that trains a model using mini-batch gradient descent
to also save the best iteration of the model using Keras library
"""


import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
    """
    Trains a model using mini-batch gradient descent,
        including analyzing validation data, using early stopping,
        and learning rate decay

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
        learning_rate_decay [boolean]:
            indicates whether learning rate decay should be used
            learning rate decay should only be performed if validation_data
            decay should be performed using inverse time decay
            learning rate should decay in a stepwise fashion after each epoch
            each time the learning rate updates, Keras should print a message
        alpha [float]:
            initial learning rate
        decay_rate [float]:
            decay rate
        save_best [boolean]:
            indicates whether to save the model after each epoch if it is best
        filepath [str]:
            file path where the model should be saved
        verbose [boolean]:
            determines if output should be printed during training
        shuffle [boolean]:
            determines whether to shuffle the batches every epoch

    returns:
        the History object generated after training the model
    """
    callback = []

    if early_stopping and validation_data:
        callback.append(
            K.callbacks.EarlyStopping(monitor='loss', patience=patience))

    def learning_rate(epoch):
        """
        calculates learning rate

        initial_learning_rate / (1 + decay_rate * (step / decay_step))
        """
        return (alpha / (1 + decay_rate * epoch))

    if learning_rate_decay and validation_data:
        callback.append(
            K.callbacks.LearningRateScheduler(learning_rate, verbose=1))

    if save_best:
        callback.append(
            K.callbacks.ModelCheckpoint(filepath=filepath,
                                        save_best_only=True))

    if callback == []:
        callback = None

    history = network.fit(x=data, y=labels,
                          batch_size=batch_size,
                          epochs=epochs,
                          validation_data=validation_data,
                          callbacks=callback,
                          verbose=verbose,
                          shuffle=shuffle)
    return history
