#!/usr/bin/env python3
"""
New code updates the script to visualize the DataFrame
"""

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove column Weighted_Price
df = df.drop(columns=['Weighted_Price'])
# Rename the column Timestamp to Date
df = df.rename(columns={'Timestamp': 'Date'})
# Convert the timestamp values to date values
df['Date'] = pd.to_datetime(df['Date'], unit='s')
df['Date'] = df['Date'].dt.to_period('d')
# The data will only plot from 2017 and beyond
df = df.loc[df['Date'] >= "2017-01-01"]
# Index the data frame on Date
df = df.set_index('Date')
# Missing values in Close should be set to the previous row value
df['Close'].fillna(method='pad', inplace=True)
# Missing values in High, Low, Open should be set to same row's Close value
df['High'].fillna(df.Close, inplace=True)
df['Low'].fillna(df.Close, inplace=True)
df['Open'].fillna(df.Close, inplace=True)
# Missing values in Volume_(BTC) and Volume_(Currency) should be set to 0
df['Volume_(BTC)'].fillna(value=0, inplace=True)
df['Volume_(Currency)'].fillna(value=0, inplace=True)
# Group values of the same day such that:
#   High: max
#   Low: min
#   Open: mean
#   Close: mean
#   Volume_(BTC): sum
#   Volume_(Currency): sum
df_plot = pd.DataFrame()
df_plot['High'] = df['High'].resample('d').max()
df_plot['Low'] = df['Low'].resample('d').min()
df_plot['Open'] = df['Open'].resample('d').mean()
df_plot['Close'] = df['Close'].resample('d').mean()
df_plot['Volume_(BTC)'] = df['Volume_(BTC)'].resample('d').sum()
df_plot['Volume_(Currency)'] = df['Volume_(Currency)'].resample('d').sum()
# Plot the data from 2017 and beyond at daily intervals
df_plot.plot()
plt.show()
