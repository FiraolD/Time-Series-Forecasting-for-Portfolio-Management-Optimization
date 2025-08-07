"""
File: task2_forecasting_models.py
Author: Financial Analyst - Guide Me in Finance (GMF) Investments
Date: August 2025
Description: Build and compare ARIMA (using statsmodels) and LSTM models to forecast TSLA prices.
            Designed to work with timezone-aware CSV data from preprocessor.py.
            No pmdarima required.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')

# -----------------------------
# 1. Load Processed Data
# -----------------------------
print("Loading processed price data...")
try:
    # Load CSV and parse index as datetime
    prices = pd.read_csv("Data/processed_prices.csv", index_col=0, parse_dates=[0])

    
    # Convert index to timezone-naive datetime, safely
    prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)

    tsla = prices[['TSLA']].copy()
    print(f"Data loaded successfully. Shape: {tsla.shape}")
    print(f"Index type: {tsla.index.dtype}")  # Should show: datetime64[ns]
except FileNotFoundError:
    raise FileNotFoundError("processed_prices.csv not found. Run preprocessor.py first.")

# -----------------------------
# 2. Train-Test Split (Chronological)
# -----------------------------
split_date = '2023-12-31'

train = tsla[tsla.index <= split_date]
test = tsla[tsla.index > split_date]

print(f"Training period: {train.index.min().date()} to {train.index.max().date()}")
print(f"Testing period: {test.index.min().date()} to {test.index.max().date()}")

# Plot train-test split
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['TSLA'], label='Training Data', color='blue')
plt.plot(test.index, test['TSLA'], label='Test Data', color='orange')
plt.axvline(pd.to_datetime(split_date), color='red', linestyle='--', label='Train/Test Split')
plt.title("TSLA Price: Train vs Test Split (Chronological)")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/train_test_split.png", dpi=300)
plt.show()

# -----------------------------
# 3. ARIMA Model (Using statsmodels)
# -----------------------------
print("\n[ARIMA] Fitting ARIMA(2,1,2) model...")
try:
    arima_model = ARIMA(train['TSLA'], order=(2, 1, 2))  # (p,d,q)
    arima_result = arima_model.fit()

    print(arima_result.summary())

    # Forecast
    n_periods = len(test)
    arima_forecast = arima_result.get_forecast(steps=n_periods)
    arima_pred_df = pd.DataFrame(arima_forecast.predicted_mean, index=test.index, columns=['ARIMA'])
    #arima_pred_df = pd.DataFrame(arima_forecast, index=test.index[:n_periods], columns=['ARIMA'])

except Exception as e:
    print(f"ARIMA model failed: {e}")
    arima_pred_df = pd.DataFrame(np.nan, index=test.index, columns=['ARIMA'])

# Plot ARIMA results
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['TSLA'], label='Train', color='blue')
plt.plot(test.index, test['TSLA'], label='Actual', color='black', linewidth=2)
if not arima_pred_df.isna().all().values:
    plt.plot(arima_pred_df.index, arima_pred_df['ARIMA'], label='ARIMA Forecast', color='green')
    # Optional: Add confidence interval
    conf_int = arima_result.get_forecast(steps=n_periods).conf_int()
    conf_int.index = test.index[:n_periods]
    plt.fill_between(conf_int.index,
                    conf_int.iloc[:, 0], conf_int.iloc[:, 1],
                    color='green', alpha=0.2, label='95% Confidence Interval')
plt.title("TSLA Price Forecast: ARIMA(2,1,2) Model")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/arima_forecast.png", dpi=300)
plt.show()

# -----------------------------
# 4. LSTM Model
# -----------------------------
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
except ImportError:
    raise ImportError("TensorFlow not installed. Run: pip install tensorflow")

# Scale data
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train)
test_scaled = scaler.transform(test)

# Create sequences
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(seq_length, len(data)):
        X.append(data[i - seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

SEQ_LENGTH = 60  # Use past 60 days

X_train, y_train = create_sequences(train_scaled, SEQ_LENGTH)
X_test, y_test = create_sequences(test_scaled, SEQ_LENGTH)

# Reshape for LSTM (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

print(f"[LSTM] Training input shape: {X_train.shape}")
print(f"[LSTM] Test input shape: {X_test.shape}")

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(SEQ_LENGTH, 1)),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25),
    Dense(1)
])

lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

# Train model
print("\n[LSTM] Training model...")
history = lstm_model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Plot training loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title("LSTM Training Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/lstm_training_loss.png", dpi=300)
plt.show()

# Make predictions
lstm_pred_scaled = lstm_model.predict(X_test)
lstm_pred = scaler.inverse_transform(lstm_pred_scaled)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create DataFrame
lstm_pred_df = pd.DataFrame(lstm_pred, index=test.index[-len(lstm_pred):], columns=['LSTM'])

# Plot LSTM results
plt.figure(figsize=(14, 6))
plt.plot(train.index, train['TSLA'], label='Train', color='blue')
plt.plot(test.index, test['TSLA'], label='Actual', color='black', linewidth=2)
plt.plot(lstm_pred_df.index, lstm_pred_df['LSTM'], label='LSTM Forecast', color='purple')
plt.title("TSLA Price Forecast: LSTM Model")
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/lstm_forecast.png", dpi=300)
plt.show()

# -----------------------------
# 5. Model Evaluation
# -----------------------------
def evaluate_model(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"Model": name, "MAE": mae, "RMSE": rmse, "MAPE (%)": mape}

# Align test data with predictions
test_aligned = test.iloc[-len(lstm_pred):]

# Evaluate ARIMA only if valid
if not arima_pred_df.isna().any().values:
    arima_eval = evaluate_model(test_aligned['TSLA'].values, arima_pred_df['ARIMA'].values, "ARIMA")
else:
    arima_eval = {"Model": "ARIMA", "MAE": np.nan, "RMSE": np.nan, "MAPE (%)": np.nan}

lstm_eval = evaluate_model(test_aligned['TSLA'].values, lstm_pred_df['LSTM'].values, "LSTM")

results = pd.DataFrame([arima_eval, lstm_eval])
print("\n📊 Model Comparison:")
print(results)

# Save evaluation
results.to_csv("outputs/model_comparison.csv", index=False)
arima_pred_df.to_csv("outputs/arima_forecast.csv")
lstm_pred_df.to_csv("outputs/lstm_forecast.csv")

print("\n✅ All forecasts, evaluations, and plots have been saved to 'outputs/' folder.")

# -----------------------------
# 6. Summary for Investment Memo
# -----------------------------
if not np.isnan(arima_eval["RMSE"]) and arima_eval["RMSE"] < lstm_eval["RMSE"]:
    best_model = "ARIMA"
else:
    best_model = "LSTM"

print(f"\n💡 Conclusion: {best_model} performed better on test data based on RMSE.")
print("This will be used to generate 6–12 month forecasts in Task 3.")