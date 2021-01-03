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
        def evaluate(self, X, Y):
            evaluates the neural network's predictions
        def gradient_descent(self, Y, cache, alpha=0.05):
            calculates one pass of gradient descent on the neural network
        def train(self, X, Y, iterations=5000, alpha=0.05,
                    verbose=True, graph=True, step=100):
            trains the neural network and updates __weights and __cache
        def save(self, filename):
            saves the instance object to a file in pickle format
        def load(filename):
            loads a pickled DeepNeuralNetwork object from a file
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
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return (prediction, cost)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        calculates one pass of gradient descent on the neural network

        parameters:
            Y [numpy.ndarray with shape (1, m)]:
                contains correct labels for the input data
            cache [dictionary]: contains intermediary values of the network,
                including X as cache[A0]
            alpha [float]: learning rate

        updates the private instance attribute __weights
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
        back = {}
        for index in range(self.L, 0, -1):
            A = cache["A{}".format(index - 1)]
            if index == self.L:
                back["dz{}".format(index)] = (cache["A{}".format(index)] - Y)
            else:
                dz_prev = back["dz{}".format(index + 1)]
                A_current = cache["A{}".format(index)]
                back["dz{}".format(index)] = (
                    np.matmul(W_prev.transpose(), dz_prev) *
                    (A_current * (1 - A_current)))
            dz = back["dz{}".format(index)]
            dW = (1 / m) * (np.matmul(dz, A.transpose()))
            db = (1 / m) * np.sum(dz, axis=1, keepdims=True)
            W_prev = self.weights["W{}".format(index)]
            self.__weights["W{}".format(index)] = (
                self.weights["W{}".format(index)] - (alpha * dW))
            self.__weights["b{}".format(index)] = (
                self.weights["b{}".format(index)] - (alpha * db))

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """
        trains the neuron and updates __weights and __cache

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
            verbose [boolean]:
                defines whether or not to print information about training
                If True, prints "Cost after {iteration} iterations: {cost}
                    after every step iterations,
                    includes data from 0th and last iteration
            graph [boolean]:
                defines whether or not to graph information about training
                If True, plots the training data every step iterations:
                    Training data is shown as a blue line,
                    X-axis is labeled as "iteration",
                    Y-axis is labeled as "cost",
                    Title of the plot is "Training Cost",
                    Includes data from the 0th and last iteration.
            step [int]: the number of iterations between printing verbose info
                    of plotting graph data point
                If verbose or graph is True:
                    If step is not int, raise TypeError.
                    If step is not positive or is greater than iterations,
                        raise ValueError.

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
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        if graph:
            import matplotlib.pyplot as plt
            x_points = np.arange(0, iterations + 1, step)
            points = []
        for itr in range(iterations):
            A, cache = self.forward_prop(X)
            if verbose and (itr % step) == 0:
                cost = self.cost(Y, A)
                print("Cost after " + str(itr) + " iterations: " + str(cost))
            if graph and (itr % step) == 0:
                cost = self.cost(Y, A)
                points.append(cost)
            self.gradient_descent(Y, cache, alpha)
        itr += 1
        if verbose:
            cost = self.cost(Y, A)
            print("Cost after " + str(itr) + " iterations: " + str(cost))
        if graph:
            cost = self.cost(Y, A)
            points.append(cost)
            y_points = np.asarray(points)
            plt.plot(x_points, y_points, 'b')
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()
        return (self.evaluate(X, Y))

    def save(self, filename):
        """
        saves the instance object to a file in pickle format

        parameters:
            filename [string]: file to save the object to
                If filename does not have extension .pkl, add it.
        """
        import pickle
        if type(filename) is not str:
            return
        if filename[-4:] != ".pkl":
            filename = filename[:] + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
            f.close()

    @staticmethod
    def load(filename):
        """
        loads a pickled DeepNeuralNetwork object from a file

        parameters:
            filename [string]: file to load object from

        returns:
            the loaded object,
                or None if filename doesn't exist
        """
        import pickle
        try:
            with open(filename, 'rb') as f:
                obj = pickle.load(f)
                return obj
        except FileNotFoundError:
            return None
