#!/usr/bin/env python3
"""
Defines a function that performs convolution
on a image with multiple channels using given padding and stride
"""


import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on images with multiple channels
    using given padding and stride

    parameters:
        images [numpy.ndarray with shape (m, h, w, c)]:
            contains multiple grayscale images
            m: number of images
            h: height in pixels of all images
            w: width in pixels of all images
            c: number of channels in the image
        kernel [numpy.ndarray with shape (kh, kw, c)]:
            contains the kernel for the convolution
            kh: height of the kernel
            kw: width of the kernel
            c: number of channels in the image
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
    m, height, width, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    if padding is 'same':
        ph = ((((height - 1) * sh) + kh - height) // 2) + 1
        pw = ((((width - 1) * sw) + kw - width) // 2) + 1
    elif padding is 'valid':
        ph = 0
        pw = 0
    else:
        ph, pw = padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)),
                    'constant', constant_values=0)
    ch = ((height + (2 * ph) - kh) // sh) + 1
    cw = ((width + (2 * pw) - kw) // sw) + 1
    convoluted = np.zeros((m, ch, cw))
    i = 0
    for h in range(0, (height + (2 * ph) - kh + 1), sh):
        j = 0
        for w in range(0, (width + (2 * pw) - kw + 1), sw):
            output = np.sum(images[:, h: h + kh, w: w + kw, :] * kernel,
                            axis=1).sum(axis=1).sum(axis=1)
            convoluted[:, i, j] = output
            j += 1
        i += 1
    return convoluted
