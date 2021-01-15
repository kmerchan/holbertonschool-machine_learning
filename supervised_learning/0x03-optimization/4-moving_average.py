#!/usr/bin/env python3
"""
Defines function that calculates the
weighted moving average of a data set
with bias correction
"""


def moving_average(data, beta):
    """
    Calculates the weighted moving average of a data set
        utilizing bias correction

    parameters:
        data [list]: data to calculate moving average of
        beta [float]: weight used for the moving average

    exponentially weighted average:
        v_t = (beta * v_(t-1)) + ((1 - beta) * sigma_t)
    bias correction:
        v_t / (1 - (beta ** t))

    returns:
        list containing the moving averages of data
    """
    v = 0
    EMA = []
    for i in range(len(data)):
        v = ((v * beta) + ((1 - beta) * data[i]))
        EMA.append(v / (1 - (beta ** (i + 1))))
    return EMA
