#!/usr/bin/env python3
"""
New code updates the script to take the last 10 columns of High and Close
   and converts them into numpy.ndarray
"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

A = # YOUR CODE HERE

print(A)
