#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create an encoder block for a transformer
"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """
    Class to create an encoder block for a transformer

    class constructor:
        def __init__(self, dm, h, hidden, drop_rate=0.1)

    public instance attribute:
        mha: MultiHeadAttention layer
        dense_hidden: the hidden dense layer with hidden units, relu activation
        dense_output: the output dense layer with dm units
        layernorm1: the first layer norm layer, with epsilon=1e-6
        layernorm2: the second layer norm layer, with epsilon=1e-6
        drouput1: the first dropout layer
        dropout2: the second dropout layer

    public instance method:
        call(self, x, training, mask=None):
            calls the encoder block and returns the block's output
    """
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """
        Class constructor

        parameters:
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads
            hidden [int]:
                represents the number of hidden units in fully connected layer
            drop_rate [float]:
                the dropout rate

        sets the public instance attributes:
            mha: MultiHeadAttention layer
            dense_hidden: the hidden dense layer with hidden units, relu activ.
            dense_output: the output dense layer with dm units
            layernorm1: the first layer norm layer, with epsilon=1e-6
            layernorm2: the second layer norm layer, with epsilon=1e-6
            drouput1: the first dropout layer
            dropout2: the second dropout layer
        """
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = None
        self.dense_output = None
        self.layernorm1 = None
        self.layernorm2 = None
        self.dropout1 = None
        self.dropout2 = None

    def call(self, x, training, mask=None):
        """
        Calls the encoder block and returns the block's output

        parameters:
            x [tensor of shape (batch, input_seq_len, dm)]:
                contains the input to the encoder block
            training [boolean]:
                determines if the model is in training
            mask:
                mask to be applied for multi-head attention

        returns:
            [tensor of shape (batch, input_seq_len, dm)]:
                contains the block's output
        """
        return None
