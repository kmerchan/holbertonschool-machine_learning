#!/usr/bin/env python3
"""
Defines function that rotates an image 90 degrees counter-clockwise
"""


import tensorflow as tf


def rotate_image(image):
    """
    Rotates an image 90 degrees counter-clockwise

    parameters:
        image [3D td.Tensor]:
            contains the image to rotate

    returns:
        the rotated image
    """
    return (tf.image.rot90(image))
