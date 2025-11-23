#!/usr/bin/env python3

import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates

from_file = __import__('2-from_file').from_file

# Read the dataset
df = from_file('datasets/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Clean the dataset
df = df.drop(columns=['Weighted_Price'])

# rename Timestamp to Date
df = df.rename(columns={"Timestamp": "Date"})

# Convert Date to datetime and set it as index
df['Date'] = pd.to_datetime(df['Date'], unit='s').dt.date

# Set Date as index
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].ffill()

# Fill missing values for High, Low, Open, Volume_(BTC), and Volume_(Currency)
df['High'] = df['High'].fillna(df['Close'])

df['Low'] = df['Low'].fillna(df['Close'])

df['Open'] = df['Open'].fillna(df['Close'])

df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)

df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter the dataset for dates after 2017-01-01
df = df[pd.to_datetime(df.index) >= '2017-01-01']

# Group by Date and aggregate the data
same_day = df.groupby(level=0).agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plot the data
plt.figure(figsize=(12, 6))

plt.plot(same_day.index, same_day['High'], label='High', color='blue', markersize=3)
plt.plot(same_day.index, same_day['Low'], label='Low', color='orange', markersize=3)
plt.plot(same_day.index, same_day['Open'], label='Open', color='green', markersize=3)
plt.plot(same_day.index, same_day['Close'], label='Close', color='red', markersize=3)

plt.plot(same_day.index, same_day['Volume_(BTC)'], label='Volume_(BTC)', color='purple', markersize=3)

plt.plot(same_day.index, same_day['Volume_(Currency)'], label='Volume_(Currency)', color='#8B4513', markersize=3)
plt.title('Date vs Close Price')
plt.legend(loc='upper right')
plt.xlabel('Date')
plt.ylabel('Close Price')

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b \n %Y'))
plt.tight_layout()
plt.savefig('Date vs Close Price.png')
plt.show()

print(same_day)
