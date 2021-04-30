#!/usr/bin/env python3
"""
Defines the class LSTMCell that represents an LSTM unit
"""


import numpy as np


class LSTMCell:
    """
    Represents a LSTM unit

    class constructor:
        def __init__(self, i, h, o)

    public instance attributes:
        Wf: forget gate weights
        bf: forget gate biases
        Wu: update gate weights
        bu: update gate biases
        Wc: intermediate cell state weights
        bc: intermediate cell state biases
        Wo: output gate weights
        bo: output gate biases
        Wy: output weights
        by: output biases

    public instance methods:
        def forward(self, h_prev, c_prev, x_t):
            performs forward propagation for one time step
    """
    def __init__(self, i, h, o):
        """
        Class constructor

        parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs

        creates public instance attributes:
            Wf: forget gate weights
            bf: forget gate biases
            Wu: update gate weights
            bu: update gate biases
            Wc: intermediate cell state weights
            bc: intermediate cell state biases
            Wo: output gate weights
            bo: output gate biases
            Wy: output weights
            by: output biases

        weights should be initialized using random normal distribution
        weights will be used on the right side for matrix multiplication
        biases should be initiliazed as zeros
        """
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Performs forward propagation for one time step

        parameters:
            h_prev [numpy.ndarray of shape (m, h)]:
                contains previous hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
            c_prev [numpy.ndarray of shape (m, h)]:
                contains previous cell state
                m: the batch size for the data
                h: dimensionality of hidden state
            x_t [numpy.ndarray of shape (m, i)]:
                contains data input for the cell
                m: the batch size for the data
                i: dimensionality of the data

        output of the cell should use softmax activation function

        returns:
            h_next, c_next, y:
            h_next: the next hidden state
            c_next: the next cell state
            y: the output of the cell
        """
        return None, None, None
