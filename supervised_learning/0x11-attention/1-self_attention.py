#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to calculate the attention for machine translation
"""


import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """
    Class to calculate the attention for machine translation

    class constructor:
        def __init__(self, units)

    public instance attribute:
        W: a Dense layer with units number of units,
            to be applied to the previous decoder hidden state
        U: a Dense layer with units number of units,
            to be applied to the encoder hidden state
        V: a Dense layer with 1 units,
            to be applied to the tanh of the sum of the outputs of W and U

    public instance method:
        def call(self, s_prev, hidden_states):
            takes in previous decoder hidden state and returns
                the context vector for decoder and the attention weights
    """
    def __init__(self, units):
        """
        Class constructor

        parameters:
            units [int]:
                represents the number of hidden units in the alignment model

        sets the public instance attributes:
            W: a Dense layer with units number of units,
                to be applied to the previous decoder hidden state
            U: a Dense layer with units number of units,
                to be applied to the encoder hidden state
            V: a Dense layer with 1 units,
                to be applied to the tanh of the sum of the outputs of W and U
        """
        if type(units) is not int:
            raise TypeError(
                "units must be int representing the number of hidden units")
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units=units)
        self.U = tf.keras.layers.Dense(units=units)
        self.V = tf.keras.layers.Dense(units=1)

    def call(self, s_prev, hidden_states):
        """
        Takes in previous decoder hidden state and outputs
            the context vector for decoder and attention weights

        parameters:
            s_prev [tensor of shape (batch, units)]:
                contains the previous decoder hidden state
            hidden_states [tensor of shape (batch, input_seq_len, units)]:
                contains the outputs of the encoder

        returns:
            context, weights:
                context [tensor of shape (batch, units)]:
                    contains the context vector for the decoder
                weights [tensor of shape (batch, input_seq_len, 1)]:
                    contains the attention weights
        """
        W = self.W(tf.expand_dims(s_prev, 1))
        U = self.U(hidden_states)
        V = self.V(tf.nn.tanh(W + U))
        weights = tf.nn.softmax(V, axis=1)
        context = tf.reduce_sum(weights * hidden_states, axis=1)
        return context, weights
