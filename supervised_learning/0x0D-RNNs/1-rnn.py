#!/usr/bin/env python3
"""
Defines function that performs forward propagation for simple RNN
"""


import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for simple RNN

    parameters:
        rnn_cell [instance of RNNCell]:
            cell that will be used for the forward propagation
        X [numpy.ndarray of shape (t, m, i)]:
            the data to be used
            t: maximum number of time steps
            m: the batch size
            i: the dimensionality of the data
        h_0 [numpy.ndarray of shape (m, h)]:
            the initial hidden state
            h: the dimensionality of the hidden state

    returns:
        H, Y:
            H [numpy.ndarray]:
                contains all the hidden states
            Y [numpy.ndarray]:
                contains all the outputs
    """
    t, m, i = X.shape
    m, h = h_0.shape
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        if step == 0:
            Y = y
        else:
            Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)
    return (H, Y)
