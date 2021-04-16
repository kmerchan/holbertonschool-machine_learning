#!/usr/bin/env python3
"""
Creates class that performs Bayesian optimization
on a noiseless 1D Gaussian process
"""


from scipy.stats import norm
import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Performs Bayesian optimization on a noiseless 1D Gaussian process

    class constructor:
        def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                     sigma_f=1, xsi=0.01, minimize=True)


    public instance attributes:
        f: the black box function
        gp: an instance of the class GaussianProcess
        X_s [numpy.ndarray of shape (ac_samples, 1)]:
            containing all acquisition sample points,
                evenly spaced between min and max
            ac_samples: number of samples
        xsi: the exploration-exploitation factor
        minimize [boolean]: for minimization versus maximization

    public instance methods:
        def acquisition(self):
            calculates the next best sample location
        def optimize(self, iterations=100):
            optimizes the black-box function
    """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Class constructor

        parameters:
            f [function]:
                the black-box function to be optimized
            X_init [numpy.ndarray of shape (t, 1)]:
                representing the inputs sampled with the black-box function
                t: number of samples
            Y_init [numpy.ndarry of shape (t, 1)]:
                representing outputs of the black-box function for each input
            bounds [tuple of (min, max)]:
                representing the bounds of the space to find the optimal point
            ac_samples [int]:
                number of samples that should be analyzed during acquisition
            l [int or float]:
                length parameter for the kernel
            sigma_f [int or float]:
                standard deviation given to output of the black-box function
            xsi [float]:
                the exploration-exploitation factor for acquisition
            minimize [boolean]:
                determines if optimization should be performed for min or max
                True: performed for minimization
                False: performed for maximization
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
        if type(bounds) is not tuple or len(bounds) != 2:
            raise TypeError("bounds must be a tuple of (min, max)")
        min, max = bounds
        if type(min) is not int and type(min) is not float:
            raise TypeError("min in bounds must be int or float")
        if type(max) is not int and type(max) is not float:
            raise TypeError("max in bounds must be int or float")
        if min >= max:
            raise ValueError("min from bounds must be less than max")
        if type(l) is not int and type(l) is not float:
            raise TypeError(
                "l must be int or float to represent kernel length parameter")
        if type(sigma_f) is not int and type(sigma_f) is not float:
            raise TypeError(
                "sigma_f must be int or float to represent standard deviation")
        if type(xsi) is not int and type(xsi) is not float:
            raise TypeError(
                "xsi must be int or float to represent \
                exploration-exploitation factor")
        if type(minimize) is not bool:
            raise TypeError("minimize must be boolean to indicate if \
            optimization should be formed for minimization or maximization")
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = X_init
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """
        Calculates the next best sample location
            using the Expected Improvement acquisition function

        returns:
            X_next, EI
            X_next [numpy.ndarray of shape (1,)]:
                represents the next best sample point
            EI [numpy.ndarray of shape (ac_samples,)]:
                contains the expected improvement of each potential sample
        """
        return None, None

    def optimize(self, iterations=100):
        """
        Optimizes the black-box function

        parameters:
            iterations [int]:
                the maximum number of iterations to perform

        If the next proposed point is one that has already been sampled,
            optimization should be stopped early.

        returns:
            X_opt, Y_opt
            X_opt [numpy.ndarray of shape (1,)]:
                representing the optimal point
            Y_opt [numpy.ndarray of shape (1,)]:
                representing the optimal function value
        """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive number")
        return None, None
