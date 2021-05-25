#!/usr/bin/env python3
"""
Defines a function that calculates the positional encoding for a transformer
"""


import numpy as np


def get_angle(pos, i, dm):
    """
    Calculates the angles for the following formulas for positional encoding:

    PE(pos, 2i) = sin(pos / 10000^(2i / dm))
    PE(pos, 2i + 1) = cos(pos / 10000^(2i / dm))
    """
    angle_rates = 1 / (10000 ** (i / dm))
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
    positional_encoding = np.zeros([max_seq_len, dm])

    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            # sin for even indices of positional_encoding
            positional_encoding[pos, i] = np.sin(get_angle(pos, i, dm))
            # cos for odd indices of positional_encoding
            positional_encoding[pos, i + 1] = np.cos(get_angle(pos, i, dm))
    return positional_encoding
