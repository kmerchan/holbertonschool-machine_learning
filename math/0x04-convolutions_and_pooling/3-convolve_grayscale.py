#!/usr/bin/env python3
"""
Defines a function that performs convolution
on a grayscale image with given padding and stride
"""


import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images with given padding and stride

    parameters:
        images [numpy.ndarray with shape (m, h, w)]:
            contains multiple grayscale images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
        kernel [numpy.ndarray with shape (kh, kw)]:
            contains the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
        padding [tuple of (ph, pw) or 'same' or 'valid']:
            ph: padding for the height of the images
            pw: padding for the width of the images
            'same' performs same convolution
            'valid' performs valid convoltuion
        stride [tuple of (sh, sw)]:
            sh: stride for the height of the image
            sw: stride for the width of the image

    if needed, images should be padded with 0s
    function may only use two for loops maximum and no other loops are allowed

    returns:
        numpy.ndarray contained convolved images
    """
    m = images.shape[0]
    height = images.shape[1]
    width = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    if padding is 'same':
        if (kh % 2) is 1:
            ph = (kh - 1) // 2
        else:
            ph = kh // 2
        if (kw % 2) is 1:
            pw = (kw - 1) // 2
        else:
            pw = kw // 2
    elif padding is 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    sh, sw = stride
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw))
    i = 0
    range_height = height + (2 * ph) - kh
    if (sh % 2) is 1:
        range_height += 1
    range_width = width + (2 * pw) - kw
    if (sw % 2) is 1:
        range_width += 1
    for h in range(0, range_height, sh):
        j = 0
        for w in range(0, range_width, sw):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, i, j] = output
            j += 1
        i += 1
    return convoluted
