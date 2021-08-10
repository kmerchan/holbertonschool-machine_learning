#!/usr/bin/env python3
"""
Defines function that randomly shears an image
"""


import tensorflow as tf


def shear_image(image, intensity):
    """
    Shears an image

    parameters:
        image [3D td.Tensor]:
            contains the image to shear
        intensity [int]:
            intensity with which the image should be sheared

    returns:
        the sheared image
    """
    image_nparray = tf.keras.preprocessing.image.img_to_array(image)
    shear_nparray = tf.keras.preprocessing.image.random_shear(image_nparray,
                                                              intensity)
    image_result = tf.keras.preprocessing.image.array_to_img(shear_nparray)
    return image_result
