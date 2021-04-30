#!/usr/bin/env python3
"""
Defines the class BidirectionalCell that represents a bidirectional RNN cell
"""


import numpy as np


class BidirectionalCell:
    """
    Represents a birectional RNN cell

    class constructor:
        def __init__(self, i, h, o)

    public instance attributes:

    public instance methods:
        def forward(self, h_prev, c_prev, x_t):
            performs forward propagation for one time step
        def backward(self, h_next, x_t):
            calculates the hidden state in backward direction for one time step
        def output(self, H):
            calculates all outputs for the RNN
    """
    def __init__(self, i, h, o):
        """
        Class constructor

        parameters:
            i: dimensionality of the data
            h: dimensionality of the hidden state
            o: dimensionality of the outputs

        creates public instance attributes:

        weights should be initialized using random normal distribution
        weights will be used on the right side for matrix multiplication
        biases should be initiliazed as zeros
        """
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wh = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step

        parameters:
            h_prev [numpy.ndarray of shape (m, h)]:
                contains previous hidden state
                m: the batch size for the data
                h: dimensionality of hidden state
            x_t [numpy.ndarray of shape (m, i)]:
                contains data input for the cell
                m: the batch size for the data
                i: dimensionality of the data

        output of the cell should use softmax activation function

        returns:
            h_next: the next hidden state
        """
        return None

    def backward(self, h_next, x_t):
        """
        calculates the hidden state in the backward direction for one time step
        """
        return None

    def output(self, H):
        """
        calculates all outputs for the RNN
        """
        return None
