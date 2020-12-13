#!/usr/bin/env python3
""" defines Poisson class that represents Poisson distribution """


class Poisson:
    """
    class that represents Poisson distribution

    class constructor:
        def __init__(self, data=None, lambtha=1.)

    instance attributes:
        lambtha [float]: the expected number of occurances in a given time

    instance methods:
        def pmf(self, k): calculates PMF for given number of successes
        def cdf(self, k): calculates CDF for given number of successes
    """

    def __init__(self, data=None, lambtha=1.):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            lambtha [float]: the expected number of occurances on a given time

        Sets the instance attribute lambtha as a float
        If data is not given:
            Use the given lambtha or
            raise ValueError if lambtha is not positive value
        If data is given:
            Calculate the lambtha of data
            Raise TypeError if data is not a list
            Raise ValueError if data does not contain at least two data points
        """
        if data is None:
            if lambtha is not >= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = lambtha
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) is not >= 2:
                raise ValueError("data must contain multiple values")
            else:
                # Lambtha will be set based on calculation from data
                # self.lambtha =
                pass

    def pmf(self, k):
        """
        calculates the value of the PMF for a given number of successes

        parameters:
            k [int]: number of successes
                If k is not an int, convert it to int
                If k is out of range, return 0

        return:
            the PMF value for k
        """
        if type(k) is not int:
            k = int(k)
        # calculates and returns the PMF

    def cdf(self, k):
        """
        calculates the value of the CDF for a given number of successes

        parameters:
            k [int]: number of successes
                If k is not an int, convert it to int
                If k is out of range, return 0

        return:
            the CDF value for k
        """
        if type(k) is not int:
            k = int(k)
        # calculates and returns the CDF
