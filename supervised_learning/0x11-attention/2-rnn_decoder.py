#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to decode for machine translation
"""


import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """
    Class to decode for machine translation

    class constructor:
        def __init__(self, vocab, embedding, units, batch)

    public instance attribute:
        embedding: a keras Enbedding layer that
            converts words from the vocabulary into an embedding vector
        gru: a keras GRU layer with units number of units
        F: a Dense layer with vocab units

    public instance methods:
        def call(self, x, s_prev, hidden_states):
            returns the output word as one hot vector and
                the new decoder hidden state
    """
    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor

        parameters:
            vocab [int]:
                represents the size of the output vocabulary
            embedding [int]:
                represents the dimensionality of the embedding vector
            units [int]:
                represents the number of hidden units in the RNN cell
            batch [int]:
                represents the batch size

        sets the public instance attributes:
            embedding: a keras Enbedding layer that
                converts words from the vocabulary into an embedding vector
            gru: a keras GRU layer with units number of units
            F: a Dense layer with vocab units
        """
        if type(vocab) is not int:
            raise TypeError(
                "vocab must be int representing the size of output vocabulary")
        if type(embedding) is not int:
            raise TypeError(
                "embedding must be int representing dimensionality of vector")
        if type(units) is not int:
            raise TypeError(
                "units must be int representing the number of hidden units")
        if type(batch) is not int:
            raise TypeError(
                "batch must be int representing the batch size")
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(input_dim=vocab,
                                                   output_dim=embedding)
        self.gru = tf.keras.layers.GRU(units=units,
                                       return_state=True,
                                       return_sequences=True,
                                       recurrent_initializer="glorot_uniform")
        self.F = tf.keras.layers.Dense(units=vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Returns the output word as a one hot vector and
            the new decoder hidden state

        parameters:
            x [tensor of shape (batch, 1)]:
                contains the previous word in the target sequence as
                    an index of the target vocabulary
            s_prev [tensor of shape (batch, units)]:
                contains the previous decoder hidden state
            hidden_states [tensor of shape (batch, input_seq_len, units)]:
                contains the outputs of the encoder

        returns:
            y, s:
                y [tensor of shape (batch, vocab)]:
                    contains the output word as a one hot vector in
                        the target vocabulary
                s [tensor of shape (batch, units)]:
                    contains the new decoder hidden state
        """
        units = s_prev.get_shape().as_list()[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)
        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)
        y, s = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)
        return y, s
