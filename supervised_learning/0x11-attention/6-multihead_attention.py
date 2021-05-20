#!/usr/bin/env python3
"""
Defines a class that inherits from tensorflow.keras.layers.Layer
to perform multi head attention
"""


sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tensorflow.keras.layers.Layer):
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
        self.h = h
        self.dm = dm
        self.depth = None
        self.Wq = None
        self.Wk = None
        self.Wv = None
        self.linear = None

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
        return None, None
