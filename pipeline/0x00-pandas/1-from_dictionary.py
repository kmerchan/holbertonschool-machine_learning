#!/usr/bin/env python3
"""
Creates a Pandas DataFrame from a dictionary and saves it into variable df
"""


import pandas as pd


def from_dictionary():
    """
    Creates a Pandas DataFrame from a dictionary

    The first column should be labeled First and have the values
        0.0, 0.5, 1.0, and 1.5.
    The second column should be labeled Second and hace the values
        one, two, three, four.
    The rows should be labeled A, B, C, and D, respectively.

    returns:
        the newly created pd.DataFrame
    """
    df = pd.DataFrame(
        {
            "First": [0.0, 0.5, 1.0, 1.5],
            "Second": ["one", "two", "three", "four"]
        },
        index=list("ABCD"))
    return df


df = from_dictionary()
