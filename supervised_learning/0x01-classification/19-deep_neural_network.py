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

    private instance attributes:
        L: the number of layers in the neural network
        cache: a dictionary holding all intermediary values of the network
        weights: a dictionary holding all weights and biases of the network

    public methods:
        def forward_prop(self, X):
            calculates the forward propagation of the neural network
        def cost(self, Y, A):
            calculates the cost of the model using logistic regression
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

        sets private instance attributes:
            __L: the number of layers in the neural network,
                initialized based on layers
            __cache: a dictionary holding all intermediary values for network,,
                initialized as an empty dictionary
            __weights: a dictionary holding all weights/biases of the network,
                weights initialized using the He et al. method
                    using the key W{l} where {l} is the hidden layer
                biases initialized to 0s
                    using the key b{l} where {1} is the hidden layer
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
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
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = weights

    @property
    def L(self):
        """
        gets the private instance attribute __L
        __L is the number of layers in the neural network
        """
        return (self.__L)

    @property
    def cache(self):
        """
        gets the private instance attribute __cache
        __cache holds all the intermediary values of the network
        """
        return (self.__cache)

    @property
    def weights(self):
        """
        gets the private instance attribute __weights
        __weights holds all the wrights and biases of the network
        """
        return (self.__weights)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples

        updates the private attribute __cache using sigmoid activation function
        sigmoid function:
            activated output = 1 / (1 + e^(-z))
            z = sum of ((__Wi * __Xi) + __b) from i = 0 to nx
        activated outputs of each layer are saved in __cache
            as A{l} where {l} is the hidden layer
        X is saved to __cache under key A0

        return:
            the output of the neural network and the cache, respectively
        """
        self.__cache["A0"] = X
        for index in range(self.L):
            W = self.weights["W{}".format(index + 1)]
            b = self.weights["b{}".format(index + 1)]
            z = np.matmul(W, self.cache["A{}".format(index)]) + b
            A = 1 / (1 + (np.exp(-z)))
            self.__cache["A{}".format(index + 1)] = A
        return (A, self.cache)

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression

        parameters:
            Y [numpy.ndarray with shape (1, m)]:
                contains correct labels for the input data
            A [numpy.ndarray with shape (1, m)]:
                contains the activated output of the neuron for each example

        logistic regression loss function:
            loss = -((Y * log(A)) + ((1 - Y) * log(1 - A)))
            To avoid log(0) errors, uses (1.0000001 - A) instead of (1 - A)
        logistic regression cost function:
            cost = (1 / m) * sum of loss function for all m example

        return:
            the calculated cost
        """
        m = Y.shape[1]
        m_loss = np.sum((Y * np.log(A)) + ((1 - Y) * np.log(1.0000001 - A)))
        cost = (1 / m) * (-(m_loss))
        return (cost)
