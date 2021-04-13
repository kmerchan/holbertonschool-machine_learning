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
            sigma_F [int or float]:
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
        self.K = self.kernel(None, None)

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
                the covariance kernal matrix between X1 and X2
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
        return None
