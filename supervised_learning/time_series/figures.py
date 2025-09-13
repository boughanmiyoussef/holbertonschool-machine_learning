#!/usr/bin/env python3
"""
Script to visualize actual vs predicted Bitcoin prices.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import pandas as pd

# preprocessed data
SEQ_LEN = 24
DATA_PATH = 'btc_preprocessed_data.npz'
MODEL_PATH = 'btc_forecast_model.keras'
OUTPUT_IMAGE = 'btc_forecast.png'

# upload data
data = np.load(DATA_PATH)
X_test = data['X_test']

# upload model
model = load_model(MODEL_PATH)

# last 24h data
last_sequence = X_test[-1].reshape(1, SEQ_LEN, 1)

# predict the next hour
predicted = model.predict(last_sequence)[0][0]

# rescale the predicted value
last_24h = last_sequence.flatten()

# concatenate the last 24h data with the predicted value
full_series = np.append(last_24h, predicted)

# shift the series for plotting
plt.figure(figsize=(12, 6))
plt.plot(range(24), last_24h, label='Actual Price (Last 24h)', color='blue')
plt.plot(24, predicted, 'ro', label='Predicted (Next Hour)', markersize=8)
plt.title('Bitcoin Price Forecast')
plt.xlabel('Hour')
plt.ylabel('Normalized Price')
plt.legend()
plt.grid(True)
plt.tight_layout()

# save the figure
plt.savefig(OUTPUT_IMAGE)
print(f"Figure saved as {OUTPUT_IMAGE}")
