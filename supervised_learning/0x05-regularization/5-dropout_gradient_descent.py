#!/usr/bin/env python3
"""
Defines function that updates the weights with Dropout regularization
using gradient descent
"""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """
    Updates the weights with Dropout regularization using gradient descent

    parameters:
        Y [one-hot numpy.ndarray of shape (classes, m)]:
            contains the correct labels for the data
            classes: number of classes
            m: number of data points
        weights [dict]:
            contains the weights and biases of the network
        cache [dict]:
            contains the outputs and dropout masks of each layer
        alpha [float]:
            learning rate
        keep_prob [float]:
            the probability that a node will be kept
        L [int]:
            number of layers in the network

    all layers should use the tanh activation function except last
    last layer should use softmax activation function

    the weights of the network should be updated in place
    """
    m = Y.shape[1]
    back = {}
    for index in range(L, 0, -1):
        A = cache["A{}".format(index - 1)]
        if index == L:
            back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
            dz = back["dz{}".format(index)]

        else:
            dz_prev = back["dz{}".format(index + 1)]
            A_current = cache["A{}".format(index)]
            back["dz{}".format(index)] = (
                np.matmul(W_prev.transpose(), dz_prev) *
                (A_current * (1 - A_current)))
            dz = back["dz{}".format(index)]
            dz *= cache["D{}".format(index)]
            dz /= keep_prob

        dW = (1 / m) * (np.matmul(dz, A.transpose()))
        db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
        W_prev = weights["W{}".format(index)]
        weights["W{}".format(index)] = (
            weights["W{}".format(index)] - (alpha * dW))
        weights["b{}".format(index)] = (
            weights["b{}".format(index)] - (alpha * db))
