#!/usr/bin/env python3
"""
Creates and trains a model to predict the value of Bitcoin
"""


import numpy as np
import pandas as pd
import tensorflow as tf


class WindowGenerator():
    """
    Creates windows, consecutive samples of data, to
        find trends within to use to make predictions
    """
    def __init__(self, input_width, label_width, shift,
                 train_data, valid_data, test_data, label_columns=None):
        """
        Class constructor
        """
        # set attributes with raw data
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data

        # label column indices
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {
                name: i for i, name in enumerate(label_columns)}
        self.column_indices = {
            name: i for i, name in enumerate(train_data.columns)}

        # window parameters
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(
            self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(
            self.total_window_size)[self.labels_slice]

    def split_window(self, features):
        """
        Converts list of consecutive inputs into window of inputs and
            window of labels
        """
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for
                 name in self.label_columns], axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    def make_dataset(self, data):
        """
        Converts time series DataFrame into tf.data.Dataset
        """
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32,)

        ds = ds.map(self.split_window)

        return ds

    @property
    def train(self):
        """
        Sets property to get tf.data.Dataset for training data
        """
        return self.make_dataset(self.train_data)

    @property
    def val(self):
        """
        Sets property to get tf.data.Dataset for validation data
        """
        return self.make_dataset(self.valid_data)

    @property
    def test(self):
        """
        Sets property to get tf.data.Dataset for testing data
        """
        return self.make_dataset(self.test_data)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


def compile_and_fit(model, window, patience=2, epochs=20):
    """
    Compiles and fits the model to return history
    """
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def time_series_forecasting(train, valid, test):
    """
    Creates and trains a model to predict value of Bitcoin

    parameters:
        train: preprocessed training dataset
        valid: preprocessed validation dataset
        test: preprocessed testing dataset
    """
    window = WindowGenerator(input_width=24, label_width=1, shift=1,
                             train_data=train, valid_data=valid,
                             test_data=test, label_columns=['Close'])

    lstm_model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(24, return_sequences=False),
        tf.keras.layers.Dense(units=1)
        ])

    history = compile_and_fit(lstm_model, window)

    val_performance = {}
    performance = {}
    val_performance['LSTM'] = lstm_model.evaluate(window.val)
    performance['LSTM'] = lstm_model.evaluate(window.test, verbose=0)


if __name__ == '__main__':
    preprocess_data = __import__('preprocess_data').preprocess_data
    train_data, valid_data, test_data = preprocess_data()
    time_series_forecasting(train_data, valid_data, test_data)
