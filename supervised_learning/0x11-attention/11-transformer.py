#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to create transformer network
"""


import tensorflow as tf
Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.layers.Layer):
    """
    Class to create the transformer network

    class constructor:
        def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                     max_seq_input, max_seq_target, drop_rate=0.1)

    public instance attributes:
        encoder: the encoder layer
        decoder: the decoder layer
        linear: the Dense layer with target_vocab units

    public instance method:
        def call(self, inputs, target, training, encoder_mask,
                    look_ahead_mask, decoder_mask):
            calls the transformer network and returns the transformer output
    """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """
        Class constructor

        parameters:
            N [int]:
                represents the number of blocks in the encoder and decoder
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads
            hidden [int]:
                represents the number of hidden units in fully connected layer
            input_vocab [int]:
                represents the size of the input vocabulary
            target_vocab [int]:
                represents the size of the target vocabulary
            max_seq_input [int]:
                represents the maximum sequence length possible for input
            max_seq_target [int]:
                represents the maximum sequence length possible for target
            drop_rate [float]:
                the dropout rate

        sets the public instance attributes:
            encoder: the encoder layer
            decoder: the decoder layer
            linear: the Dense layer with target_vocab units
        """
        if type(N) is not int:
            raise TypeError(
                "N must be int representing number of blocks in the encoder")
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        if type(hidden) is not int:
            raise TypeError(
                "hidden must be int representing number of hidden units")
        if type(input_vocab) is not int:
            raise TypeError(
                "input_vocab must be int representing size of input vocab")
        if type(target_vocab) is not int:
            raise TypeError(
                "target_vocab must be int representing size of target vocab")
        if type(max_seq_input) is not int:
            raise TypeError(
                "max_seq_input must be int representing max length for input")
        if type(max_seq_target) is not int:
            raise TypeError(
                "max_seq_target must be int representing max len for target")
        if type(drop_rate) is not float:
            raise TypeError(
                "drop_rate must be float representing dropout rate")
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(units=target_vocab)

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """
        Calls the transformer network and returns the transformer output

        parameters:
            inputs [tensor of shape (batch, input_seq_len)]:
                contains the inputs
            target [tensor of shape (batch, target_seq_len)]:
                contains the target
            training [boolean]:
                determines if the model is in training
            encoder_mask:
                padding mask to be applied to the encoder
            look_ahead_mask:
                look ahead mask to be applied to the decoder
            decoder_mask:
                padding mask to be applied to the decoder

        returns:
            [tensor of shape (batch, target_seq_len, target_vocab)]:
                contains the transformer output
        """
        encoder_output = self.encoder(inputs, training, encoder_mask)
        decoder_output = self.decoder(target, encoder_output, training,
                                      look_ahead_mask, decoder_mask)
        final_output = self.linear(decoder_output)
        return final_output
