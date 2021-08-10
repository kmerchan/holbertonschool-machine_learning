#!/usr/bin/env python3
"""
Defines function that changes the hue of an image
"""


import tensorflow as tf


def change_hue(image, delta):
    """
    Changes the hue of an image

    parameters:
        image [3D td.Tensor]:
            contains the image to change
        delta [float]:
            the amount the hue should change

    returns:
        the altered image
    """
    return (tf.image.adjust_hue(image, delta))
