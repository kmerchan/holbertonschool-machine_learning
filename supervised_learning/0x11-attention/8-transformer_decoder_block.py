#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create a decoder block for a transformer
"""


import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class DecoderBlock(tf.keras.layers.Layer):
    """
    Class to create a decoder block for a transformer

    class constructor:
        def __init__(self, dm, h, hidden, drop_rate=0.1)

    public instance attribute:
        mha1: the first MultiHeadAttention layer
        mha2: the second MultiHeadAttention layer
        dense_hidden: the hidden dense layer with hidden units, relu activation
        dense_output: the output dense layer with dm units
        layernorm1: the first layer norm layer, with epsilon=1e-6
        layernorm2: the second layer norm layer, with epsilon=1e-6
        layernorm3: the third layer norm layer, with epsilon=1e-6
        drouput1: the first dropout layer
        dropout2: the second dropout layer
        dropout3: the third dropout layer

    public instance method:
        def call(self, x, encoder_output, training, look_ahead_mask,
                    padding_mask):
            calls the decoder block and returns the block's output
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
            mha1: the first MultiHeadAttention layer
            mha2: the second MultiHeadAttention layer
            dense_hidden: the hidden dense layer with hidden units, relu activ.
            dense_output: the output dense layer with dm units
            layernorm1: the first layer norm layer, with epsilon=1e-6
            layernorm2: the second layer norm layer, with epsilon=1e-6
            layernorm3: the third layer norm layer, with epsilon=1e-6
            drouput1: the first dropout layer
            dropout2: the second dropout layer
            dropout3: the third dropout layer
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
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(units=hidden,
                                                  activation='relu')
        self.dense_output = tf.keras.layers.Dense(units=dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask, padding_mask):
        """
        Calls the decoder block and returns the block's output

        parameters:
            x [tensor of shape (batch, target_seq_len, dm)]:
                contains the input to the decoder block
            encoder_output [tensor of shape (batch, input_seq_len, dm)]:
                contains the output of the encoder
            training [boolean]:
                determines if the model is in training
            look_ahead_mask:
                mask to be applied to the first multi-head attention
            padding_mask:
                mask to be applied to the second multi-head attention

        returns:
            [tensor of shape (batch, target_seq_len, dm)]:
                contains the block's output
        """
        attention_output1, _ = self.mha1(x, x, x, look_ahead_mask)
        attention_output1 = self.dropout1(attention_output1, training=training)
        output1 = self.layernorm1(x + attention_output1)

        attention_output2, _ = self.mha2(output1, encoder_output,
                                         encoder_output, padding_mask)
        attention_output2 = self.dropout2(attention_output2, training=training)
        output2 = self.layernorm2(output1 + attention_output2)

        dense_output = self.dense_hidden(output2)
        ffn_output = self.dense_output(dense_output)
        ffn_output = self.dropout3(ffn_output, training=training)
        output3 = self.layernorm3(output2 + ffn_output)

        return output3
