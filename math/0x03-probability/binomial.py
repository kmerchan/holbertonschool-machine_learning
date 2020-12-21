#!/usr/bin/env python3
""" defines Binomial class that represents binomial distribution """


class Binomial:
    """
    class that represents Binomial distribution

    class constructor:
        def __init__(self, data=None, n=1, p=0.5)

    instance attributes:
        n [int]: the number of Bernoilli trials
        p [float]: the probability of a success

    instance methods:
        def pmf(self, k): calculates PMF for given number of successes
        def cdf(self, k): calculates CDF for given number of successes
    """

    def __init__(self, data=None, n=1, p=0.5):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            n [int]: the number of Bernoilli trials
            p [float]: the probability of a success

        Sets the instance attributes n and p
        If data is not given:
            Use the given n and p
            Raise ValueError if n is not positive value
            Raise ValueError if p is not a valid probability
        If data is given:
            Calculate n and p from data, rounding n to nearest int
            Raise TypeError if data is not a list
            Raise ValueError if data does not contain at least two data points
        """
        if data is None:
            if n < 1:
                raise ValueError("n must be a positive value")
            else:
                self.n = n
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            else:
                self.p = p
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                mean = float(sum(data) / len(data))
                summation = 0
                for x in data:
                    summation += ((x - mean) ** 2)
                variance = (summation / len(data))
                q = variance / mean
                p = (1 - q)
                n = round(mean / p)
                p = float(mean / n)
                self.n = n
                self.p = p

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
        if k < 0:
            return 0
        p = self.p
        n = self.n
        q = (1 - p)
        n_factorial = 1
        for i in range(n):
            n_factorial *= (i + 1)
        k_factorial = 1
        for i in range(k):
            k_factorial *= (i + 1)
        nk_factorial = 1
        for i in range(n - k):
            nk_factorial *= (i + 1)
        binomial_co = n_factorial / (k_factorial * nk_factorial)
        pmf = binomial_co * (p ** k) * (q ** (n - k))
        return pmf

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
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
