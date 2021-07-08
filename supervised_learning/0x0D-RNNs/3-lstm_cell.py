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
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))
        self.Wf = np.random.normal(size=(h + i, h))
        self.Wu = np.random.normal(size=(h + i, h))
        self.Wc = np.random.normal(size=(h + i, h))
        self.Wo = np.random.normal(size=(h + i, h))
        self.Wy = np.random.normal(size=(h, o))

    def softmax(self, x):
        """
        Performs the softmax function

        parameters:
            x: the value to perform softmax on to generate output of cell

        return:
            softmax of x
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        softmax = e_x / e_x.sum(axis=1, keepdims=True)
        return softmax

    def sigmoid(self, x):
        """
        Performs the sigmoid function

        parameters:
            x: the value to perform sigmoid on

        return:
            sigmoid of x
        """
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid

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
        concatenation = np.concatenate((h_prev, x_t), axis=1)
        u_gate = self.sigmoid
