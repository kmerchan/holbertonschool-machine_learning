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

    public instance attributes:
        W1: the weights vector for the hidden layer
        b1: the bias for the hidden layer
        A1: the activated output for the hidden layer
        W2: the weights vector for the output neuron
        b2: the bias for the output neuron
        A2: the activated output for the output neuron
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

        sets public instance attributes:
            W1: the weights vector for the hidden layer,
                initialized using a random normal distribution
            b1: the bias for the hidden layer,
                initialized with 0s
            A1: the activated output for the hidden layer,
                initialized to 0
            W2: the weights vector for the output neuron,
                initialized using a random normal distribution
            b2: the bias for the output neuron,
                initialized to 0
            A2: the activated output for the output neuron,
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
        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = 0
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
