#!/usr/bin/env python3
"""
Defines function that creates a confusion matrix
from given labels and predictions
"""


def create_confusion_matrix(labels, logits):
    """
    Creates a confusion matrix from given labels and prediction

    parameters:
        labels [numpy.ndarray of shape (m, classes)]:
            contains the correct labels of each data point
            m: number of data points
            classes: number of classes
        logits [numpy.ndarray of shape (m, classes)]:
            contains the predicted labels of each data point
            m: number of data points
            classes: number of classes

    returns:
        confusion matrix [numpy.ndarray of shape (classes, classes)
            with row indices representing correct labels
            and column indices representing predicted labels
    """
    return np.matmul(labels.transpose(), logits)
