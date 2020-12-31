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

    private instance attributes:
        __W: the weights vector for the neuron
        __b: the bias for the neuron
        __A: the activated output of the neuron (prediction)

    public methods:
        def forward_prop(self, X):
            calculates the forward propagation of the neuron
        def cost(self, Y, A):
            calculates the cost of the model using logistic regression
        def evaluate(self, X, Y):
            evaluates the neuron's predictions
    """

    def __init__(self, nx):
        """
        class constructor

        parameters:
            nx [int]: the number of input features to the neuron
            If nx is not an integer, raise a TypeError.
            If nx is less than 1, raise a ValueError.

        sets private instance attributes:
            __W: the weights vector for the neuron,
                initialized using a random normal distribution
            __b: the bias for the neuron,
                initialized to 0
            __A: the activated output of the neuron (prediction),
                initialized to 0
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """
        gets the private instance attribute __W
        __W is the weights vector for the neuron
        """
        return (self.__W)

    @property
    def b(self):
        """
        gets the private instance attribute __b
        __b is the bias for the neuron
        """
        return (self.__b)

    @property
    def A(self):
        """
        gets the private instance attribute __A
        __A is the activated output of the neuron
        """
        return (self.__A)

    def forward_prop(self, X):
        """
        calculates the forward propagation of the neuron

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples

        updates the private attribute __A using sigmoid activation function
        sigmoid function:
            __A = 1 / (1 + e^(-z))
            z = sum of ((__Wi * __Xi) + __b) from i = 0 to nx

        return:
            the updated private attribute __A
        """
        z = np.matmul(self.W, X) + self.b
        self.__A = 1 / (1 + (np.exp(-z)))
        return (self.A)

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

    def evaluate(self, X, Y):
        """
        evaluates the neuron's predictions

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y [numpy.ndarray with shape (1, m)]:
                contains correct labels for the input data

        returns:
            the neuron's prediction and the cost of the network, respectively
            prediction is numpy.ndarray with shape (1, m), containing
                predicted labels for each example
            label values should be 1 if the output of the network is >= 0.5,
                0 if the output of the network is < 0.5
        """
        A = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)
