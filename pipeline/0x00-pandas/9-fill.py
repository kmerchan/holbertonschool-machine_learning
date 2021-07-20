#!/usr/bin/env python3
"""
New code fills in the missing data points in the DataFrame
"""

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove column Weighted_Price
df = df.drop(columns=['Weighted_Price'])
# Missing values in Close should be set to previous row value
df['Close'].fillna(method='pad', inplace=True)
# Missing values in High, Low, Open should be set to same row's Close value
df['High'].fillna(df.Close, inplace=True)
df['Low'].fillna(df.Close, inplace=True)
df['Open'].fillna(df.Close, inplace=True)
# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)

print(df.head())
print(df.tail())
