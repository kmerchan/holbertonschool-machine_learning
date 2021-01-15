#!/usr/bin/env python3
"""
Defines function that updates the learning rate
using inverse time decay in numpy
"""


import numpy as np


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    Updates the learning rate using inverse time decay in numpy

    parameters:
        alpha [float]: original learning rate
        decay_rate: wight used to determine the rate at which alpha will decay
        global_step [int]:
            number of passes of gradient descent that have elapsed
        decay_step [int]:
            number of passes of gradient descent that should occur before
                alpha is decayed furtherXS

    the learning rate decay should occur in a stepwise fashion

    returns:
        the updated value for alpha
    """
