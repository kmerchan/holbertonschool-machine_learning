#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to encode for machine translation
"""


import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Class to encode for machine translation

    class constructor:
        def __init__(self, vocab, embedding, units, batch)

    public instance attribute:
        batch: the batch size
        units: the number of hidden units in the RNN cell
        embedding: a keras Enbedding layer that
            converts words from the vocabulary into an embedding vector
        gru: a keras GRU layer with units number of units

    public instance methods:
        def initialize_hidden_state(self):
            initializes the hidden states for the RNN cell to a tensor of zeros
        def call(self, x, initial):
            takes in initial hidden state and returns outputs of the encoder
                and last hidden state of the encoder
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        parameters:
            vocab [int]:
                represents the size of the input vocabulary
            embedding [int]:
                represents the dimensionality of the embedding vector
            units [int]:
                represents the number of hidden units in the RNN cell
            batch [int]:
                represents the batch size

        sets the public instance attributes:
            batch: the batch size
            units: the number of hidden units in the RNN cell
            embedding: a keras Enbedding layer that
                converts words from the vocabulary into an embedding vector
            gru: a keras GRU layer with units number of units
        """
        if type(vocab) is not int:
            raise TypeError(
                "vocab must be int representing the size of input vocabulary")
        if type(embedding) is not int:
            raise TypeError(
                "embedding must be int representing dimensionality of vector")
        if type(units) is not int:
            raise TypeError(
                "units must be int representing the number of hidden units")
        if type(batch) is not int:
            raise TypeError(
                "batch must be int representing the batch size")
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer="glorot_uniform")

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of zeros

        returns:
            [tensor of shape (batch, units)]:
                containing the initialized hidden states
        """
        hidden_states = tf.zeros(shape=(self.batch, self.units))
        return hidden_states

    def call(self, x, initial):
        """
        Calls the encoder with given input to encoder layer and returns output

        parameters:
            x [tensor of shape (batch, input_seq_len)]:
                contains the input to the encoder layer as word indices
                    within the vocabulary
            initial [tensor of shape (batch, units)]:
                contains the initial hidden state

        returns:
            outputs, hidden:
                outputs [tensor of shape (batch, input_seq_len, units)]:
                    contains the outputs of the encoder
                hidden [tensor of shape (batch, units)]:
                    contains the last hidden state of the encoder
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
