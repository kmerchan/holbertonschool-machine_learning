#!/usr/bin/env python3
"""
Defines a function that creates a confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix

    parameters:
        labels [numpy.ndarray of shape (m, classes)]:
            contains correct labels for each data point
            m: number of data points
            classes: number of classes
        logits [numpy.ndarray of shape (m, classes)]:
            contains predicted labels for each data point
            m: number of data points
            classes: number of classes

    returns:
        confusion matrix [numpy.ndarray of shape (classes, classes)]:
            row indices represent correct labels
            column indices represent predicted labels
    """
    return np.matmul(labels.transpose(), logits)
