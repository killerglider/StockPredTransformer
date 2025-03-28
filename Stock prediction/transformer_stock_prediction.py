import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# List of stocks
stocks = ["AAPL", "AMZN", "BLK", "BAC", "FDX", "JNJ", "NVDA", "MSFT", "GOOG", "UPS"]

# Function to load stock data (for simplicity, using yfinance)
import yfinance as yf

def load_stock_data(stock, start="2020-01-01", end="2024-01-01"):
    data = yf.download(stock, start=start, end=end)
    return data["Close"].values.reshape(-1, 1)

# Load data and scale
scaler = MinMaxScaler()
dataset = {stock: scaler.fit_transform(load_stock_data(stock)) for stock in stocks}

# Function to create sequences
def create_sequences(data, seq_length=60):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 60
X_train, y_train = create_sequences(dataset["AAPL"], seq_length)  # Example with AAPL

# Transformer Model
def build_transformer_model(input_shape):
    inputs = keras.Input(shape=input_shape)
    x = layers.LSTM(128, return_sequences=True)(inputs)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dense(32, activation="relu")(x)
    x = layers.Dense(16, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

# Model training
model = build_transformer_model((seq_length, 1))
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.1)

# Predictions
predicted = model.predict(X_train)

# Rescale predictions and actual values
y_train_rescaled = scaler.inverse_transform(y_train.reshape(-1, 1))
predicted_rescaled = scaler.inverse_transform(predicted)

# Evaluation
mse = mean_squared_error(y_train_rescaled, predicted_rescaled)
rmse = np.sqrt(mse)
r2 = r2_score(y_train_rescaled, predicted_rescaled)
mae = mean_absolute_error(y_train_rescaled, predicted_rescaled)
accuracy = 1 - (mae / np.mean(y_train_rescaled))

print(f"MSE: {mse}, RMSE: {rmse}, R2 Score: {r2}, MAE: {mae}, Accuracy: {accuracy}")

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(y_train_rescaled, label="Actual Prices")
plt.plot(predicted_rescaled, label="Predicted Prices")
plt.legend()
plt.title("Stock Price Prediction using Transformer Model")
plt.show()
