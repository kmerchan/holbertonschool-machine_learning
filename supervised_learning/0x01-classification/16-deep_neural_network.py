#!/usr/bin/env python3
"""
defines DeepNeuralNetwork class that defines
a deep neural network performing binary classification
"""


import numpy as np


class DeepNeuralNetwork:
    """
    class that represents a deep neural network
    performing binary classification

    class constructor:
        def __init__(self, nx, layers)

    public instance attributes:
        L: the number of layers in the neural network
        cache: a dictionary holding all intermediary values of the network
        weights: a dictionary holding all weights and biases of the network
    """

    def __init__(self, nx, layers):
        """
        class constructor

        parameters:
            nx [int]: the number of input features
                If nx is not an integer, raise a TypeError.
                If nx is less than 1, raise a ValueError.
            layers [list]: representing the number of nodes in each layer
                If layers is not a list, raise TypeError.
                If elements in layers are not all positive ints,
                    raise a TypeError.

        sets public instance attributes:
            L: the number of layers in the neural network,
                initialized based on layers
            cache: a dictionary holding all intermediary values of the network,
                initialized as an empty dictionary
            weights: a dictionary holding all weights & biases of the network,
                weights initialized using the He et al. method
                    using the key W{l} where {l} is the hidden layer
                biases initialized to 0s
                    using the key b{l} where {1} is the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list:
            raise TypeError("layers must be a list of positive integers")
        weights = {}
        previous = nx
        for index, layer in enumerate(layers, 1):
            if type(layer) is not int or layer < 0:
                raise TypeError("layers must be a list of positive integers")
            weights["b{}".format(index)] = np.zeros((layer, 1))
            weights["W{}".format(index)] = (
                np.random.randn(layer, previous) * np.sqrt(2 / previous))
            previous = layer
        self.L = len(layers)
        self.cache = {}
        self.weights = weights
