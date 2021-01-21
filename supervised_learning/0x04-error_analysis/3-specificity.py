#!/usr/bin/env python3
"""
Defines function that calculates the specificity
for each class in a confusion matrix
"""


import numpy as np


def specificity(confusion):
    """
    Calculates the specificity for each class in a confusion matrix

    parameters:
        confusion [numpy.ndarray of shape (classes, classes)]:
            confusion matrix where row indices represent the correct labels
            and column indices represent the predicted labels

    returns:
        numpy.ndarray of shape (classes,) containing specificity of each class
    """
    classes = confusion.shape[0]
    specificity = []
    for actual_class in range(classes):
        true_negative = 0
        total = 0
        for row in range(classes):
            if row == actual_class:
                continue
            for column in range(classes):
                if column != actual_class:
                    true_negative += confusion[row][column]
                total += confusion[row][column]
        specificity.append(true_negative / total)
    return np.asarray(specificity)
