#!/usr/bin/env python3
"""
Preprocess Bitcoin price data for time series forecasting.
This script loads Bitcoin price data from two sources,combines them,
and prepares the data for training a time series forecasting model.
It includes the following steps:
1. Load the data from CSV files.
2. Remove rows with missing values.
3. Convert the 'Timestamp' column from UNIX timestamp to datetime.
4. Combine the two datasets.
5. Keep only the 'Timestamp' and 'Close' columns.
6. Set 'Timestamp' as the DataFrame index.
7. Resample the data to hourly frequency, taking the last price per hour.
8. Scale the 'Close' prices between 0 and 1 for model training.
9. Create sequences of past 24 hours of data for training.
10. Split the data into training, validation, and test sets.
11. Save the preprocessed data to a .npz file.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Sequence length (e.g., 24 hours)
SEQ_LEN = 24


# Function to create sequences for time series data
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)


# Load datasets
bitstamp = pd.read_csv(
    '../data/bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv'
)

coinbase = pd.read_csv(
    '../data/coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv'
)

# Remove rows with missing value
bitstamp.dropna(inplace=True)
coinbase.dropna(inplace=True)

# Convert 'Timestamp' column from UNIX timestamp to datetime
bitstamp['Timestamp'] = pd.to_datetime(bitstamp['Timestamp'], unit='s')
coinbase['Timestamp'] = pd.to_datetime(coinbase['Timestamp'], unit='s')

# Combine the two datasets
combined = pd.concat([bitstamp, coinbase], ignore_index=True)

# Keep only 'Timestamp' and 'Close' columns
combined = combined[['Timestamp', 'Close']]
combined.dropna(inplace=True)

# Set 'Timestamp' as the DataFrame index
combined.set_index('Timestamp', inplace=True)

# Resample data to hourly frequency, taking the last price per hour
data_by_hour = combined.resample('1h').last().dropna()

# Scale the 'Close' prices between 0 and 1 for model training
scaler = MinMaxScaler()
data_by_hour['Close'] = scaler.fit_transform(data_by_hour[['Close']])


# Show the first 5 rows of the processed data
print(data_by_hour.head())

# Create sequences
close_values = data_by_hour['Close'].values
X, y = create_sequences(close_values, SEQ_LEN)

# Split data (no shuffle)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.2, shuffle=False
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, shuffle=False
)

# Save
np.savez('btc_preprocessed_data.npz',
         X_train=X_train, y_train=y_train,
         X_val=X_val, y_val=y_val,
         X_test=X_test, y_test=y_test)

# Select the last 24 hours of data to plot
last_24h = data_by_hour.tail(24)

# Plot normalized Bitcoin closing price for the last 24 hours
plt.figure(figsize=(14, 6))
plt.plot(
    last_24h.index, last_24h['Close'], label='BTC Close Price (Last 24 Hours)'
)
plt.title('Bitcoin Close Price - Last 24 Hours')
plt.xlabel('Time')
plt.ylabel('Normalized Close Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
