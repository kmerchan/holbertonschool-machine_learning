#!/usr/bin/env python3
"""
Defines a function that performs forward propagation
over a pooling layer of a neural network
"""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs forward propagation over pooling layer of a neural network

    parameters:
        A_prev [numpy.ndarray of shape (m, h_prev, w_prev, c_prev)]:
            contains the output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        kernel_shape [tuple of shape (kh, kw)]:
            contains size of kernel for pooling
            kh: filter height
            kw: filter width
        stride [tuple of shape (sh, sw)]:
            contains the strides for the pooling
            sh: stride for the height
            sw: stride for the width
        mode [string: 'max' or 'avg']:
            indicates whether to perform max of average pooling

    returns:
        output of the pooling layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = ((h_prev - kh) // sh) + 1
    pw = ((w_prev - kw) // sw) + 1
    pooled = np.zeros((m, ph, pw, c_prev))
    i = 0
    for h in range(0, (h_prev - kh + 1), sh):
        j = 0
        for w in range(0, (w_prev - kw + 1), sw):
            if mode == 'max':
                output = np.max(A_prev[:, h:h + kh, w:w + kw, :],
                                axis=(1, 2))
            elif mode == 'avg':
                output = np.average(A_prev[:, h:h + kh, w:w + kw, :],
                                    axis=(1, 2))
            else:
                pass
            pooled[:, i, j, :] = output
            j += 1
        i += 1
    return pooled
