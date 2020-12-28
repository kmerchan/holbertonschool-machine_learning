#!/usr/bin/env python3
"""
defines Neuron class that defines
a single neuron performing binary classification
"""


import numpy as np


class Neuron:
    """
    class that represents a single neuron performing binary classification

    class constructor:
        def __init__(self, nx)

    public instance attributes:
        W: the weights vector for the neuron
        b: the bias for the neuron
        A: the activated output of the neuron (prediction)
    """

    def __init__(self, nx):
        """
        class constructor

        parameters:
            nx [int]: the number of input features to the neuron
            If nx is not an integer, raise a TypeError.
            If nx is less than 1, raise a ValueError.

        sets public instance attributes:
            W: the weights vector for the neuron,
                initialized using a random normal distribution
            b: the bias for the neuron,
                initialized to 0
            A: the activated output of the neuron (prediction),
                initialized to 0
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
