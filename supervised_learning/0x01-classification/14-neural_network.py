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
        def cost(self, Y, A):
            calculates the cost of the model using logistic regression
        def evaluate(self, X, Y):
            evaluates the neural network's predictions
        def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
            calculates one pass of gradient descent on the neural network
        def train(self, X, Y, iterations=5000, alpha=0.05):
            trains the neural network and updates
                __W1, __b1, __A1, __W2, __b2, and __A2
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
        evaluates the neural network's predictions

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
        A1, A2 = self.forward_prop(X)
        cost = self.cost(Y, A2)
        prediction = np.where(A2 >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """
        calculates one pass of gradient descent on the neural network

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y [numpy.ndarray with shape (1, m)]:
                contains correct labels for the input data
            A1 [numpy.ndarray with shape (1, m)]:
                 contains the activated output of the hidden layer
            A2 [numpy.ndarray with shape (1, m)]:
                 contains the predicted output
            alpha [float]: learning rate

        updates the private instance attributes __W1, __b1, __W2, and __b2
            using back propagation

        derivative of loss function with respect to A:
            dA = (-Y / A) + ((1 - Y) / (1 - A))
        derivative of A with respect to z:
            dz = A * (1 - A)
        combining two above with chain rule,
        derivative of loss function with respect to z:
            dz = A - Y
        using chain rule with above derivative,
        derivative of loss function with respect to __W:
            d__Wi = Xidz or vectorized as d__W = (1 / m) * (dz dot X transpose)
        derivative of loss function with respect to __b:
            d__bi = dz of vectorized as d__b = (1 / m) * (sum of dz elements)

        for neural network, using the derivatives above:
        derivative of loss function with respect to z2:
            dz2 = A2 - Y
        derivative of loss function with respect to __W2:
            d__W2 = (1 / m) * (dz1 dot A1 transpose)
        derivative of loss function with respect to __b2:
            d__b2 = (1 / m) * (sum of dz2 over axis 1)
        derivative of loss function with respect to z1:
            dz1 = (__W2 transpose dot dz2) * A1(1 - A1)
        derivative of loss function with respect to __W1:
            d__W1 = (1 / m) * (dz dot X transpose)

        one-step of gradient descent updates the attributes with the following:
            __W = __W - (alpha * d__W)
            __b = __b - (alpha * d__b)
        """
        m = Y.shape[1]
        dz2 = (A2 - Y)
        d__W2 = (1 / m) * (np.matmul(dz2, A1.transpose()))
        d__b2 = (1 / m) * (np.sum(dz2, axis=1, keepdims=True))
        dz1 = (np.matmul(self.W2.transpose(), dz2)) * (A1 * (1 - A1))
        d__W1 = (1 / m) * (np.matmul(dz1, X.transpose()))
        d__b1 = (1 / m) * (np.sum(dz1, axis=1, keepdims=True))
        self.__W1 = self.W1 - (alpha * d__W1)
        self.__b1 = self.b1 - (alpha * d__b1)
        self.__W2 = self.W2 - (alpha * d__W2)
        self.__b2 = self.b2 - (alpha * d__b2)

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        trains the neuron and updates __W1, __b1, __A1, __W2, __b2, and __A2

        parameters:
            X [numpy.ndarray with shape (nx, m)]: contains the input data
                nx is the number of input features to the neuron
                m is the number of examples
            Y [numpy.ndarray with shape (1, m)]:
                contains correct labels for the input data
            iterations [int]: the number of iterations to train over
                If iterations is not an int, raise TypeError.
                If iterations is not positive, raise ValueError.
            alpha [float]: learning rate
                If alpha is not an int, raise TypeError.
                If alpha is not positive, raise ValueError.

        returns:
            the evaluation of the training data after iterations of training
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        for itr in range(iterations):
            A1, A2 = self.forward_prop(X)
            self.gradient_descent(X, Y, A1, A2, alpha)
        return (self.evaluate(X, Y))
