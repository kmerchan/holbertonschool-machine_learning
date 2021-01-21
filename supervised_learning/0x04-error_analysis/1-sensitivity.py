#!/usr/bin/env python3
"""
Defines function that calculates the sensitivity
for each class in a confusion matrix
"""


import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix

    parameters:
        confusion [numpy.ndarray of shape (classes, classes)]:
            confusion matrix where row indices represent the correct labels
            and column indices represent the predicted labels

    returns:
        numpy.ndarray of shape (classes,) containing sensitivity of each class
    """
    classes = confusion.shape[0]
    sensitivity = []
    for row in range(classes):
        correct = 0
        total = 0
        for column in range(classes):
            if row == column:
                correct += confusion[row][column]
            total += confusion[row][column]
        sensitivity.append(correct / total)
    return np.asarray(sensitivity)
