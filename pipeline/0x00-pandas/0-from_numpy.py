#!/usr/bin/env python3
"""
Defines function that creates a Pandas DataFrame from a Numpy ndarray
"""


import pandas as pd


def from_numpy(array):
    """
    Creates a Pandas DataFrame from a numpy.ndarray

    parameters:
        array [numpy.ndarray]: array to create pd.DataFrame from

    columns of the DataFrame should be labeled in alphabetical order
        and capitalized (there will not be more than 26 columns)

    returns:
        the newly created pd.DataFrame
    """
    alphabet = ["A", "B", "C", "D", "E", "F", "G", "H", "I",
                "J", "K", "L", "M", "N", "O", "P", "Q", "R",
                "S", "T", "U", "V", "W", "X", "Y", "Z"]
    column_labels = []
    for i in range(len(array[0])):
        column_labels.append(alphabet[i])
    df = pd.DataFrame(array, columns=column_labels)
    return df
