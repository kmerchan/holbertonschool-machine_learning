#!/usr/bin/env python3
"""
Defines function that performs a random crop of an image
"""


import tensorflow as tf


def crop_image(image, size):
    """
    Performs a random crop of an image

    parameters:
        image [3D td.Tensor]:
            contains the image to crop
        size [tuple]:
            contains the size of the crop

    returns:
        the cropped image
    """
    return (tf.image.random_crop(image, crop_size=size))
