#!/usr/bin/env python3
"""
defines NeuralNetwork class that defines
a neural network with one hidden layer
performing binary classification
"""


import numpy as np


class NeuralNetwork:
    """
    class that represents a neural network with one hidden layer
    performing binary classification

    class constructor:
        def __init__(self, nx, nodes)

    private instance attributes:
        __W1: the weights vector for the hidden layer
        __b1: the bias for the hidden layer
        __A1: the activated output for the hidden layer
        __W2: the weights vector for the output neuron
        __b2: the bias for the output neuron
        __A2: the activated output for the output neuron

    public methods:
        def forward_prop(self, X):
            calculates the forward propagation of the neural network
    """

    def __init__(self, nx, nodes):
        """
        class constructor

        parameters:
            nx [int]: the number of input features
                If nx is not an integer, raise a TypeError.
                If nx is less than 1, raise a ValueError.
            nodes [int]: the number of nodes found in the hidden layer
                If nodes is not an integer, raise TypeError.
                If nodes is less than 1, raise a ValueError.

        sets private instance attributes:
            __W1: the weights vector for the hidden layer,
                initialized using a random normal distribution
            __b1: the bias for the hidden layer,
                initialized with 0s
            __A1: the activated output for the hidden layer,
                initialized to 0
            __W2: the weights vector for the output neuron,
                initialized using a random normal distribution
            __b2: the bias for the output neuron,
                initialized to 0
            __A2: the activated output for the output neuron,
                initialized to 0
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """
        gets the private instance attribute __W1
        __W1 is the weights vector for the hidden layern
        """
        return (self.__W1)

    @property
    def b1(self):
        """
        gets the private instance attribute __b1
        __b1 is the bias for the hidden layer
        """
        return (self.__b1)

    @property
    def A1(self):
        """
        gets the private instance attribute __A1
        __A1 is the activated output of the hidden layer
        """
        return (self.__A1)

    @property
    def W2(self):
        """
        gets the private instance attribute __W2
        __W2 is the weights vector for the output neuron
        """
        return (self.__W2)

    @property
    def b2(self):
        """
        gets the private instance attribute __b2
        __b2 is the bias for the output neuron
        """
        return (self.__b2)

    @property
    def A2(self):
        """
        gets the private instance attribute __A2
        __A2 is the activated output of the output neuron
        """
        return (self.__A2)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neural network

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples

        updates the private attributes __A1 and __A2
            using sigmoid activation function
        sigmoid function:
            __A = 1 / (1 + e^(-z))
            z = sum of ((__Wi * __Xi) + __b) from i = 0 to nx

        return:
            the updated private attributes __A1 and __A2, respectively
        """
        z1 = np.matmul(self.W1, X) + self.b1
        self.__A1 = 1 / (1 + (np.exp(-z1)))
        z2 = np.matmul(self.W2, self.__A1) + self.b2
        self.__A2 = 1 / (1 + (np.exp(-z2)))
        return (self.A1, self.A2)
