#!/usr/bin/env python3
"""
Defines class MultiNormal that represents a Multivariate Normal Distribution
"""


import numpy as np


class MultiNormal:
    """
    Class that represents Multivariate Normal Distribution

    class constructor:
        def __init__(self, data)

    public instance variables:
        mean [numpy.ndarray of shape (d, 1)]:
            contains the mean of data
        cov [numpy.ndarray of shape (d, d)]:
            contains the covariance matrix of data

    public instance method:
        def pdf(self, x):
            calculates the PDF at a data point
    """
    def __init__(self, data):
        """
        Class constructor

        parameters:
            data [numpy.ndarray of shape (d, n)]:
                contains the data set
                d: number of dimensions in each data point
                n: number of data points
        """
        if type(data) is not np.ndarray or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, axis=1, keepdims=True)
        self.mean = mean
        cov = np.matmul(data - mean, data.T - mean.T) / (n - 1)
        self.cov = cov

    def pdf(self, x):
        """
        Calculates the PDF at a data point

        parameters:
            x [numpy.ndarray of shape (d, 1)]:
                contains the data point whose PDF should be calculated
                d: number of dimensions of the instance

        returns:
            value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        test_d, one = x.shape
        if test_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf
