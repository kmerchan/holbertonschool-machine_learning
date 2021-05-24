#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to perform multi head attention
"""


import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    Class to perform multi-head attention

    class constructor:
        def __init__(self, dm, h)

    public instance attribute:
        h: number of heads
        dm: the dimensionality of the model
        depth: the depth of each attention head
        Wq: a Dense layer with dm units, used to generate the query matrix
        Wk: a Dense layer with dm units, used to generate the key matrix
        Wv: a Dense layer with dm units, used to generate the value matrix
        linear: a Dense layer with dm units, used to generate attention output

    public instance methods:
        def call(self, Q, K, V, mask):
            generates the query, key, and value matrices and
                outputs the scaled dot product attention
    """
    def __init__(self, dm, h):
        """
        Class constructor

        parameters:
            dm [int]:
                represents the dimensionality of the model
            h [int]:
                represents the number of heads

        sets the public instance attributes:
            h: number of heads
            dm: the dimensionality of the model
            depth: the depth of each attention head
            Wq: a Dense layer with dm units, used to generate the query matrix
            Wk: a Dense layer with dm units, used to generate the key matrix
            Wv: a Dense layer with dm units, used to generate the value matrix
            linear: a Dense layer with dm units,
                used to generate attention output
        """
        if type(dm) is not int:
            raise TypeError(
                "dm must be int representing dimensionality of model")
        if type(h) is not int:
            raise TypeError(
                "h must be int representing number of heads")
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h
        self.Wq = tf.keras.layers.Dense(units=dm)
        self.Wk = tf.keras.layers.Dense(units=dm)
        self.Wv = tf.keras.layers.Dense(units=dm)
        self.linear = tf.keras.layers.Dense(units=dm)

    def split_heads(self, x, batch):
        """
        Splits the last dimension of tensor into (h, dm) and
            transposes the result so the shape is (batch, h, seq_len, dm)
        """
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def call(self, Q, K, V, mask):
        """
        Generates the query, key, and value matrices and
            outputs the scaled dot product attention

        parameters:
            Q [tensor of shape (batch, seq_len_q, dk)]:
                contains the input to generate the query matrix
            K [tensor of shape (batch, seq_len_v, dk)]:
                contains the input to generate the key matrix
            V [tensor of shape (batch, seq_len_v, dv)]:
                contains the input to generate the value matrix
            mask [always None]

        returns:
            outputs, weights:
                outputs [tensor with last two dimensions (..., seq_len_q, dm)]:
                    contains the scaled dot product attention
                weights [tensor with last dimensions
                        (..., h, seq_len_q, seq_len_v)]:
                    contains the attention weights
        """
        # batch = Q.get_shape().as_list()[0]
        batch = tf.shape(Q)[0]

        q = self.Wq(Q)
        k = self.Wk(K)
        v = self.Wv(V)

        q = self.split_heads(q, batch)
        k = self.split_heads(k, batch)
        v = self.split_heads(v, batch)

        attention, weights = sdp_attention(q, k, v, mask)

        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch, -1, self.dm))
        outputs = self.linear(concat_attention)

        return outputs, weights
