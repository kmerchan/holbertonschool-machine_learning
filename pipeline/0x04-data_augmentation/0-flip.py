#!/usr/bin/env python3
"""
Defines function that flips an image horizontally
"""


import tensorflow as tf


def flip_image(image):
    """
    Flips an image horizontally

    parameters:
        image [3D td.Tensor]:
            contains the image to flip

    returns:
        the flipped image
    """
    return (tf.image.flip_left_right(image))
