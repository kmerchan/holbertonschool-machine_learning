#!/usr/bin/env python3
"""
Defines a function that performs forward propagation
over a convolutional neural network
"""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Performs forward propagation over a convolutional neural network

    parameters:
        A_prev [numpy.ndarray of shape (m, h_prev, w_prev, c_prev)]:
            contains the output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        W [numpy.ndarray of shape(kh, kw, c_prev, c_new)]:
            contains the kernels for the convolution
            kh: filter height
            kw: filter width
            c_prev: number of channels in the previous layer
            c_new: number of channels in the output
        b [numpy.ndarray of shape (1, 1, 1, c_new)]:
            contains the biases applied to the convolution
        activation [function]:
            activation function applied to the convolution
        padding [string: 'same' or 'valid']:
            indicates the type of padding used for the convolution
        stride [tuple of shape (sh, sw)]:
            contains the strides for the convolution
            sh: stride for the height
            sw: stride for the width

    returns:
        output of the convolutional layer
    """
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride
    if padding is 'valid':
        ph = 0
        pw = 0
    elif padding is 'same':
        ph = ((((h_prev - 1) * sh) + kh - h_prev) // 2)
        pw = ((((w_prev - 1) * sw) + kw - w_prev) // 2)
    else:
        return
    images = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((h_prev + (2 * ph) - kh) // sh) + 1
    cw = ((w_prev + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw, c_new))
    for index in range(c_new):
        kernel_index = W[:, :, :, index]
        i = 0
        for h in range(0, (h_prev + (2 * ph) - kh + 1), sh):
            j = 0
            for w in range(0, (w_prev + (2 * pw) - kw + 1), sw):
                output = np.sum(
                    images[:, h:h + kh, w:w + kw, :] * kernel_index,
                    axis=1).sum(axis=1).sum(axis=1)
                output += b[0, 0, 0, index]
                convoluted[:, i, j, index] = activation(output)
                j += 1
            i += 1
    return convoluted
