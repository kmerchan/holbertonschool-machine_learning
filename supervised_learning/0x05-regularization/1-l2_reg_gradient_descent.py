#!/usr/bin/env python3
"""
Defines a function that updates the weights and biases
using gradient descent with L2 Regularization
"""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Updates weights and biases using gradient descent with L2 regularization

    parameters:
        Y [one-hot numpy.ndarray of shape (classes, m)]:
            contains the correct labels for the data
            classes: number of classes
            m: number of data points
        weights [dict]: dictionary of weights and biases for the network
        cache [dict]: dictionary of the outputs of each layer of the network
        alpha [float]: learning rate
        lambtha: the regularization parameter
        L: the number of layers in the neural network

    Neural network using tanh activations on each layer except the last.
    Last layer uses softmax activation.
    """
    m = Y.shape[1]
    back = {}
    for index in range(L, 0, -1):
        A = cache["A{}".format(index - 1)]
        if index == L:
            back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
        else:
            dz_prev = back["dz{}".format(index + 1)]
            A_current = cache["A{}".format(index)]
            back["dz{}".format(index)] = (
                np.matmul(W_prev.transpose(), dz_prev) *
                (A_current * (1 - A_current)))
        dz = back["dz{}".format(index)]
        dW = (1 / m) * (
            (np.matmul(dz, A.transpose())) + (
                lambtha * weights["W{}".format(index)]))
        db = (1 / m) * (
            (np.sum(dz, axis=1, keepdims=True)) + (
                lambtha * weights["b{}".format(index)]))
        W_prev = weights["W{}".format(index)]
        weights["W{}".format(index)] = (
            weights["W{}".format(index)] - (alpha * dW))
        weights["b{}".format(index)] = (
            weights["b{}".format(index)] - (alpha * db))
