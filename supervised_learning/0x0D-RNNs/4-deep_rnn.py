#!/usr/bin/env python3
"""
Defines function that performs forward propagation for a deep RNN
"""


import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    Performs forward propagation for a deep RNN
    parameters:
        rnn_cells [list of instances of RNNCell]:
            cells that will be used for the forward propagation
            l: list length; number of layers in deep RNN
        X [numpy.ndarray of shape (t, m, i)]:
            the data to be used
            t: maximum number of time steps
            m: the batch size
            i: the dimensionality of the data
        h_0 [numpy.ndarray of shape (l, m, h)]:
            the initial hidden state
            l: number of layers
            m: the batch size
            h: the dimensionality of the hidden state

    returns:
        H, Y:
            H [numpy.ndarray]:
                contains all the hidden states
            Y [numpy.ndarray]:
                contains all the outputs
    """
    layers = len(rnn_cells)
    t, m, i = X.shape
    l, m, h = h_0.shape
    H = np.zeros((t + 1, layers, m, h))
    H[0] = h_0
    for step in range(t):
        for layer in range(layers):
            if layer == 0:
                h_prev = X[step]
            h_prev, y = rnn_cells[layer].forward(H[step, layer], h_prev)
            H[step + 1, layer, ...] = h_prev
            if layer == layers - 1:
                if step == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))
    output_shape = Y.shape[-1]
    Y = Y.reshape(t, m, output_shape)
    return (H, Y)
