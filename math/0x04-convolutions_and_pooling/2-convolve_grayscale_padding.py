#!/usr/bin/env python3
"""
Defines a function that performs convolution with custom padding
on a grayscale image
"""


import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
    Performs a convolution on grayscale images with custom padding

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
        padding [tuple of (ph, pw)]:
            ph: padding for the height of the images
            pw: padding for the width of the images

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
    ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)),
                    'constant', constant_values=0)
    ch = height + (2 * ph) - kh + 1
    cw = width + (2 * pw) - kw + 1
    convoluted = np.zeros((m, ch, cw))
    for h in range(ch):
        for w in range(cw):
            output = np.sum(images[:, h: h + kh, w: w + kw] * kernel,
                            axis=1).sum(axis=1)
            convoluted[:, h, w] = output
    return convoluted
