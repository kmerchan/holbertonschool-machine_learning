#!/usr/bin/env python3
"""
Defines function that randomly changes the brightness of an image
"""


import tensorflow as tf


def change_brightness(image, max_delta):
    """
    Randomly changes the brightness of an image

    parameters:
        image [3D td.Tensor]:
            contains the image to change
        max_delta [float]:
            maximum amount the image should be brightened (or darkened)

    returns:
        the altered image
    """
    return (tf.image.random_brightness(image, max_delta))
