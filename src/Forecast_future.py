"""
File: task3_forecast_future.py
Author: Financial Analyst - GMF Investments
Date: August 2025
Description: Use the trained LSTM model to forecast TSLA prices 6–12 months ahead.
            Fixed scaling issue to avoid unrealistic predictions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# 1. Load Data and Trained LSTM
# -----------------------------
print("Loading data and trained LSTM model...")

# Load processed data
prices = pd.read_csv("Data/processed_prices.csv", index_col=0, parse_dates=[0])
#prices.index = pd.to_datetime(prices.index).tz_localize(None)
prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)

tsla = prices[['TSLA']].copy()

# Load model and scaler
lstm_model = load_model("outputs/lstm_model.h5")
scaler = joblib.load("outputs/scaler.pkl")

print("✅ Model and scaler loaded.")

# -----------------------------
# 2. Prepare Last Sequence (Scaled)
# -----------------------------
SEQ_LENGTH = 60
last_sequence = tsla[-SEQ_LENGTH:].values
last_sequence_scaled = scaler.transform(last_sequence)

X_input = last_sequence_scaled.reshape((1, SEQ_LENGTH, 1))

# -----------------------------
# 3. Forecast 250 Days (12 months)
# -----------------------------
forecast_horizon = 250
forecasted_scaled = []

for _ in range(forecast_horizon):
    pred = lstm_model.predict(X_input, verbose=0)
    forecasted_scaled.append(pred[0, 0])

    # Update sequence in scaled space
    X_input = np.roll(X_input, -1, axis=1)
    X_input[0, -1, 0] = pred[0, 0]

# Inverse transform to get real prices
forecasted_scaled = np.array(forecasted_scaled).reshape(-1, 1)
forecasted_prices = scaler.inverse_transform(forecasted_scaled)

# Create future dates
last_date = tsla.index[-1]
future_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)

# Create DataFrame
forecast_df = pd.DataFrame(forecasted_prices, index=future_dates, columns=['Forecast'])

# -----------------------------
# 4. Plot Forecast
# -----------------------------
plt.figure(figsize=(14, 7))
plt.plot(tsla.index[-200:], tsla['TSLA'][-200:], label='Historical TSLA', color='blue')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='LSTM Forecast (12 months)', color='orange', linestyle='--')
plt.title("TSLA Price Forecast: 12-Month Outlook")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/future_forecast.png", dpi=300)
plt.show()

# -----------------------------
# 5. Save Forecast
# -----------------------------
forecast_df.to_csv("outputs/future_forecast.csv")
projected_price = forecasted_prices[-1, 0]
current_price = tsla['TSLA'].iloc[-1]
expected_return = (projected_price - current_price) / current_price * 100

print(f"✅ Forecast saved. Expected price on {future_dates[-1].date()}: ${projected_price:.2f}")
print("\n📈 Forecast Summary:")
print(f"Current TSLA Price: ${current_price:.2f}")
print(f"Projected Price (12mo): ${projected_price:.2f}")
print(f"Expected Return: {expected_return:.2f}%")
print("⚠️  Model uncertainty increases over time — long-term forecasts should be used cautiously.")