#!/usr/bin/env python3
""" defines Normal class that represents normal distribution """


class Normal:
    """
    class that represents normal distribution

    class constructor:
        def __init__(self, data=None, mean=0., stddev=1.)

    instance attributes:
        mean [float]: the mean of the distribution
        stddev [float]: the standard deviation of the distribution

    instance methods:
        def z_score(self, x): calculates the z-score of a given x-value
        def x_value(self, z): calculates the x-value of a given z-score
        def pdf(self, x): calculates PDF for given x-value
        def cdf(self, x): calculates CDF for given x-value
    """

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        class constructor

        parameters:
            data [list]: data to be used to estimate the distibution
            mean [float]: the mean of the distribution
            stddev [float]: the standard deviation of the distribution

        Sets the instance attributes mean and stddev as floats
        If data is not given:
            Use the given mean and stddev
            raise ValueError if stddev is not positive value
        If data is given:
            Calculate the mean and stddev of data
            Raise TypeError if data is not a list
            Raise ValueError if data does not contain at least two data points
        """
        if data is None:
            if stddev is not >= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.stddev = stddev
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            elif len(data) is not >= 2:
                raise ValueError("data must contain multiple values")
            else:
                # calculates mean and stddev of data
                pass

    def z_score(self, x):
        """
        calculates the z-score of a given x-value

        parameters:
            x: x-value

        return:
            z-score of x
        """
        pass

    def x_value(self, z):
        """
        calculates the x-value of a given z-score

        parameters:
            z: z-score

        return:
            x-value of z
        """
        pass

    def pdf(self, x):
        """
        calculates the value of the PDF for a given x-value

        parameters:
            x: x-value

        return:
            the PDF value for x
        """
        # calculates and returns the PDF
        pass

    def cdf(self, x):
        """
        calculates the value of the CDF for a given x-value

        parameters:
            x: x-value

        return:
            the CDF value for x
        """
        # calculates and returns the CDF
        pass
