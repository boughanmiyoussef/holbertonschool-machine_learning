#!/usr/bin/env python3
"""
Train and validate an RNN model for forecasting BTC price using past 24 hours.
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Load preprocessed data
data = np.load('btc_preprocessed_data.npz')
X_train, y_train = data['X_train'], data['y_train']
X_val, y_val = data['X_val'], data['y_val']
X_test, y_test = data['X_test'], data['y_test']

# Create tf.data.Datasets
BATCH_SIZE = 64
train_dataset = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).batch(BATCH_SIZE)
val_dataset = tf.data.Dataset.from_tensor_slices(
    (X_val, y_val)
    ).batch(BATCH_SIZE)

# Build model
model = tf.keras.Sequential([
    layers.Input(shape=(X_train.shape[1], 1)),
    layers.LSTM(64, return_sequences=False),
    layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train
history = model.fit(train_dataset,
                    validation_data=val_dataset,
                    epochs=10)

# Save model
model.save('btc_forecast_model.keras')
