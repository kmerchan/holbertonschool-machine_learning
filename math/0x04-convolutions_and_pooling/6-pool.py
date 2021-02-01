#!/usr/bin/env python3
"""
Defines a function that performs pooling on images
"""


import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
    Performs pooling on images

    parameters:
        images [numpy.ndarray with shape (m, h, w, c)]:
            contains multiple images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
            c: number of channels in the image
        kernel_shape [tuple of (kh, kw)]:
            contains the kernel shape for pooling
            kh: height of the kernel
            kw: width of the kernel
        stride [tuple of (sh, sw)]:
            sh: stride for the height of the image
            sw: stride for the width of the image
        mode [string: 'max' or 'avg']:
            indicates the type of pooling

    function may only use two for loops maximum and no other loops are allowed

    returns:
        numpy.ndarray contained pooled images
    """
    m, height, width, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    ph = ((height - kh) // sh) + 1
    pw = ((width - kw) // sw) + 1
    pooled = np.zeros((m, ph, pw, c))
    i = 0
    for h in range(0, (height - kh + 1), sh):
        j = 0
        for w in range(0, (width - kw + 1), sw):
            if mode == 'max':
                output = np.max(images[:, h:h + kh, w:w + kw, :],
                                axis=(1, 2))
            elif mode == 'avg':
                output = np.average(images[:, h:h + kh, w:w + kw, :],
                                    axis=(1, 2))
            else:
                pass
            pooled[:, i, j, :] = output
            j += 1
        i += 1
    return pooled
