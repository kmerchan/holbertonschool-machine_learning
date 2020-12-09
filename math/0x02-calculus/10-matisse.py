#!/usr/bin/env python3
""" defines a function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """
    calculates the derivative of the given polynomial

    Parameters:
        poly (list): list of coefficients representing a polynomial
            the index of the list represents the power of x
            the coefficient belongs to

    Returns:
        a new list of coefficients representing the derivative
        [0], if the derivate is 0
        None, if poly is not valid
    """
    if type(poly) is not list or len(poly) < 1:
        return None
    for coefficient in poly:
        if type(coefficient) is not int and type(coefficient) is not float:
            return None
    for power, coefficient in enumerate(poly):
        if power is 0:
            derivative = [0]
            continue
        if power is 1:
            derivative = []
        derivative.append(power * coefficient)
    while derivative[-1] is 0 and len(derivative) > 1:
        derivative = derivative[:-1]
    return derivative
