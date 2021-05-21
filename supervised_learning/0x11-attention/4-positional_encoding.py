#!/usr/bin/env python3
"""
Defines a function that calculates the positional encoding for a transformer
"""


import numpy as np


def get_angles(pos, i, dm):
    """
    Calculates the angles for the following formulas for positional encoding:

    PE(pos, 2i) = sin(pos / 10000^(2i / dm))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / dm))
    """
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """
    Calculates the positional encoding for a transformer

    parameters:
        max_seq_len [int]:
            represents the maximum sequence length
        dm: model depth

    returns:
        [numpy.ndarray of shape (max_seq_len, dm)]:
            contains the positional encoding vectors
    """
    angle_rads = get_angles(np.arrange(max_len_seq)[:, np.newaxis],
                            np.arrange(dm)[np.newaxis, :],
                            dm)
    # apply sin to every 2i, even indices of angle_rads
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    # apply cos to every 2i + 1, odd indices of angle_rads
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    positional_encoding = tf.cast(angle_rads[np.newaxis, ...],
                                  dtype=tf.float32)
    return positional_encoding
