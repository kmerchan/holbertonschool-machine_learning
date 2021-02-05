#!/usr/bin/env python3
"""
Defines a function that performs backward propagation
over a pooling layer of a neural network
"""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Performs backward propagation over a pooling layer of neural network

    parameters:
        dA [numpy.ndarray of shape (m, h_new, w_new, c)]:
            contains the partial derivatives with respect to the
                output of the pooling layer
            m: number of examples
            h_new: height of the output
            w_new: weight of the output
            c: number of channels in the output
        A_prev [numpy.ndarray of shape (m, h_prev, w_prev, c)]:
            contains the output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c: number of channels
        kernel_shape [tuple of shape (kh, kw)]:
            contains size of kernal for pooling
            kh: filter height
            kw: filter width
        stride [tuple of shape (sh, sw)]:
            contains the strides for the pooling
            sh: stride for the height
            sw: stride for the width
        mode [string: 'max' or 'avg']:
            indicates whether to perform max of average pooling

    returns:
        the partial derivative with respect to the previous layer (dA_prev)
    """
    m, h_new, w_new, c = dA.shape
    m, h_prev, w_prev, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros((m, h_prev, w_prev, c))
    for ex in range(m):
        for kernel_index in range(c):
            for h in range(h_new):
                for w in range(w_new):
                    i = h * sh
                    j = w * sw
                    if mode is 'max':
                        pool = A_prev[ex, i: i + kh, j: j + kw, kernel_index]
                        mask = np.where(pool == np.max(pool), 1, 0)
                    elif mode is 'avg':
                        mask = np.ones((kh, kw))
                        mask /= (kh * kw)
                    dA_prev[ex, i: i + kh, j: j + kw, kernel_index] += (
                        mask * dA[ex, h, w, kernel_index])
    return dA_prev
