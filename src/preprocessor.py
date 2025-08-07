"""
File: preprocessor.py
Description: Robustly download and preprocess financial data for TSLA, BND, SPY.
Handles network timeouts and retries.
"""

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import requests
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



# -----------------------------
# 2. Define tickers and date range
# -----------------------------
tickers = ['TSLA', 'BND', 'SPY']
start_date = "2015-07-01"
end_date = "2025-07-31"

# -----------------------------
# 3. Download data one by one with retry logic
# -----------------------------
def download_ticker(symbol, retries=3, delay=2):
    for i in range(retries):
        try:
            print(f"Downloading {symbol}... (Attempt {i+1}/{retries})")
            ticker = yf.Ticker(symbol)
            data = ticker.history(start=start_date, end=end_date)
            if data.empty:
                raise Exception("No data returned")
            print(f"✓ Successfully downloaded {symbol}")
            return data
        except Exception as e:
            print(f"Failed to download {symbol}: {e}")
            if i < retries - 1:
                print(f"Retrying in {delay} seconds...")
                time.sleep(delay + np.random.uniform(1, 3))  # Add jitter
            else:
                print(f"❌ Failed to download {symbol} after {retries} attempts.")
                return None

# Dictionary to store data
raw_data = {}
for ticker in tickers:
    raw_data[ticker] = download_ticker(ticker )
    time.sleep(0.5)  # Gentle pause between tickers

# -----------------------------
# 4. Combine into single DataFrame
# -----------------------------
# Extract Adjusted Close only
adj_close = {}
for ticker, df in raw_data.items():
    if df is not None and 'Close' in df.columns:
        adj_close[ticker] = df['Close']  # Use 'Close' if 'Adj Close' not available
        # Or use: adj_close[ticker] = df['Adj Close'] if you want adjusted prices
    else:
        print(f"⚠️ Warning: Missing data for {ticker}")

# Convert to DataFrame
prices = pd.DataFrame(adj_close)
prices.index = pd.to_datetime(prices.index)
prices.sort_index(inplace=True)

print(f"\nFinal data shape: {prices.shape}")
print(f"Date range: {prices.index.min()} to {prices.index.max()}")

# Save to CSV as backup
prices.to_csv("Data/processed_prices.csv")
print("✅ Data saved to 'processed_prices.csv'")

# -----------------------------
# 5. Proceed with EDA only if data is valid
# -----------------------------
if prices.isna().all().all():
    print("❌ No valid data available. Exiting.")
else:
    # Fill missing values (e.g., due to different trading days)
    prices = prices.interpolate().bfill()

    print("\nBasic statistics:")
    print(prices.describe())

    # Plot prices
    plt.figure(figsize=(14, 8))
    for col in prices.columns:
        plt.plot(prices.index, prices[col], label=col)
    plt.title("Adjusted Close Prices (2015–2025)")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("price_trends.png", dpi=300)
    plt.show()

    # Continue with returns, volatility, etc., as before...
    returns = prices.pct_change().dropna()
    print("\nDaily returns calculated.")