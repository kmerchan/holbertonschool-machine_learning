#!/usr/bin/env python3
"""
Defines function that performs PCA color augmentation
"""


import tensorflow as tf


def pca_color(image, alphas):
    """
    Perfoms PCA color augmentation on an image

    parameters:
        image [3D td.Tensor]:
            contains the image to change
        alphas [tuple of length 3]:
            contains the amount that each channel should change

    returns:
        the augmented image
    """
    image_nparray = tf.keras.preprocessing.image.img_to_array(image)

    # flattens to RGB channels as floats
    img = image_nparray.reshape(-1, 3).astype(float)

    return img
