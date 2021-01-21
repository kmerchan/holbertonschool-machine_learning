#!/usr/bin/env python3
"""
Defines function that calculates the F1 score
for each class in a confusion matrix
"""


import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculates the F1 score for each class in a confusion matrix

    parameters:
        confusion [numpy.ndarray of shape (classes, classes)]:
            confusion matrix where row indices represent the correct labels
            and column indices represent the predicted labels

    returns:
        numpy.ndarray of shape (classes,) containing F1 score of each class
    """
    p = precision(confusion)
    r = sensitivity(confusion)
    F1 = (2 * p * r) / (p + r)
    return F1
