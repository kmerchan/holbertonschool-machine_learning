#!/usr/bin/env python3
"""
New code transposes the rows and columns and then sorts the data
    in reverse chronological order
"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.sort_values(by='Timestamp', ascending=False).T

print(df.tail(8))
