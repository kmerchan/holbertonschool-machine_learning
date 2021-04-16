#!/usr/bin/env python3
"""
Creates class that represents a noiseless 1D Gaussian process
"""


import numpy as np


class GaussianProcess:
    """
    Represents a noiseless 1D Gaussian process

    class constructor:
        def __init__(self, X_init, Y_init, l=1, sigma_f=1)

    public instance attributes:
        X [numpy.ndarray of shape (t, 1)]:
            representing the inputs sampled with the black-box function
            t: number of samples
        Y [numpy.ndarry of shape (t, 1)]:
            representing the outputs of the black-box function for each input
        l [int]:
            length parameter for the kernel
        sigma_F [float]:
            standard deviation given to the output of the black-box function
        K [numpy.ndarray]:
            representing the current covariance kernel matrix

    public instance method:
        def kernel(self, X1, X2):
            calculates the covariance kernel matrix between two matrices
        def predict(self, X_s):
            predicts mean and standard deviation of points in Gaussian process
        def update(self, X_new, Y_new):
            updates the Gaussian process
    """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """
        Class constructor

        parameters:
            X_init [numpy.ndarray of shape (t, 1)]:
                representing the inputs sampled with the black-box function
                t: number of samples
            Y_init [numpy.ndarry of shape (t, 1)]:
                representing outputs of the black-box function for each input
            l [int or float]:
                length parameter for the kernel
            sigma_f [int or float]:
                standard deviation given to output of the black-box function
        """
        if type(X_init) is not np.ndarray or len(X_init.shape) != 2:
            raise TypeError("X_init must be numpy.ndarray of shape (t, 1)")
        t, one = X_init.shape
        if one != 1:
            raise TypeError("X_init must be numpy.ndarray of shape (t, 1)")
        if type(Y_init) is not np.ndarray or len(Y_init.shape) != 2:
            raise TypeError("Y_init must be numpy.ndarray of shape (t, 1)")
        t_check, one = Y_init.shape
        if one != 1 or t_check != t:
            raise TypeError("Y_init must be numpy.ndarray of shape (t, 1)")
        if type(l) is not int and type(l) is not float:
            raise TypeError(
                "l must be int or float to represent kernel length parameter")
        if type(sigma_f) is not int and type(sigma_f) is not float:
            raise TypeError(
                "sigma_f must be int or float to represent standard deviation")
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """
        Calculates the covariance kernel matrix between two matrices

        parameters:
            X1 [numpy.ndarray of shape (m, 1)]:
                first matrix with m number of samples
            X2 [numpy.ndarray of shape (n, 1)]:
                second matrix with n number of samples

        The kernel should use the Radial Basis Function (RBF)

        returns:
            [numpy.ndarray of shape (m, n)]:
                the covariance kernel matrix between X1 and X2
        """
        if type(X1) is not np.ndarray or len(X1.shape) != 2:
            raise TypeError("X1 must be numpy.ndarray of shape (m, 1)")
        m, one = X1.shape
        if one != 1:
            raise TypeError("X1 must be numpy.ndarray of shape (m, 1)")
        if type(X2) is not np.ndarray or len(X2.shape) != 2:
            raise TypeError("X2 must be numpy.ndarray of shape (n, 1)")
        n, one = X2.shape
        if one != 1:
            raise TypeError("X2 must be numpy.ndarray of shape (n, 1)")
        X1_sum = np.sum(X1 ** 2, 1).reshape(-1, 1)
        X2_sum = np.sum(X2 ** 2, 1)
        sqdist = X1_sum + X2_sum - 2 * np.matmul(X1, X2.T)
        cov = (self.sigma_f ** 2) * np.exp(-0.5 / (self.l ** 2) * sqdist)
        return cov

    def predict(self, X_s):
        """
        Predicts mean and standard deviation of points in a Gaussian process

        parameters:
            X_s [numpy.ndarray of shape (s, 1)]:
                contains all the points whose mean and standard deviation
                    should be calculated
                s: number of sample points

        returns:
            mu, sigma
                mu [numpy.ndarray of shape (s,)]:
                    contains the mean for each point in X_s
                sigma [numpy.ndarray of shape (s,)]:
                    contains the variance for each point in X_s
        """
        if type(X_s) is not np.ndarray or len(X_s.shape) != 2:
            raise TypeError("X_s must be numpy.ndarray of shape (s, 1)")
        s, one = X_s.shape
        if one != 1:
            raise TypeError("X_s must be numpy.ndarray of shape (s, 1)")
        K = self.K
        K_s = self.kernel(self.X, X_s)
        K_inv = np.linalg.inv(K)
        K_ss = self.kernel(X_s, X_s)
        mu_s = K_s.T.dot(K_inv).dot(self.Y)
        mu = mu_s.reshape(-1)
        cov_s = K_ss - K_s.T.dot(K_inv).dot(K_s)
        sigma = np.diag(cov_s)
        return mu, sigma

    def update(self, X_new, Y_new):
        """
        Updates the Gaussian process

        parameters:
            X_new [numpy.ndarray of shape (1,)]:
                represents the new sample point
            Y_new [numpy.ndarray of shape (1,)]:
                represents the new same function value

        Updates the public instance attributes X, Y, and K
        """
        if type(X_new) is not np.ndarray or X_new.shape[0] != 1:
            raise TypeError("X_new must be numpy.ndarray of shape (1,)")
        if type(Y_new) is not np.ndarray or Y_new.shape[0] != 1:
            raise TypeError("Y_new must be numpy.ndarray of shape (1,)")
        self.X = np.append(self.X, X_new)[:, np.newaxis]
        self.Y = np.append(self.Y, Y_new)[:, np.newaxis]
        self.K = self.kernel(self.X, self.X)
