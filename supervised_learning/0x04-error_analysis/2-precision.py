#!/usr/bin/env python3
"""
Defines function that calculates the precision
for each class in a confusion matrix
"""


import numpy as np


def precision(confusion):
    """
    Calculates the precision for each class in a confusion matrix

    parameters:
        confusion [numpy.ndarray of shape (classes, classes)]:
            confusion matrix where row indices represent the correct labels
            and column indices represent the predicted labels

    returns:
        numpy.ndarray of shape (classes,) containing precision of each class
    """
    classes = confusion.shape[0]
    precision = []
    for column in range(classes):
        correct = 0
        total = 0
        for row in range(classes):
            if row == column:
                correct += confusion[row][column]
            total += confusion[row][column]
        precision.append(correct / total)
    return np.asarray(precision)
