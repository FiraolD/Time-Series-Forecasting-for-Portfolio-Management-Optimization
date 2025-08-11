"""
Forward-Looking Portfolio Optimization
Combines LSTM forecast for TSLA with historical or forecasted data for BND & SPY
"""

import pandas as pd
import numpy as np
from pypfopt import expected_returns, risk_models, EfficientFrontier

print("Loading forecast and price data...")

# --- 1. Load historical processed prices ---
hist_prices = pd.read_csv("Data/processed_prices.csv", index_col=0, parse_dates=True)

# --- 2. Load TSLA forecast (already realistic) ---
tsla_forecast = pd.read_csv("outputs/future_forecast.csv", index_col=0, parse_dates=True)
tsla_forecast.rename(columns={"Forecast": "TSLA"}, inplace=True)

# --- 3. Build forward-looking prices ---
# For BND and SPY, take last price and assume small growth (or replace with real forecasts)
last_bnd_price = hist_prices["BND"].iloc[-1]
last_spy_price = hist_prices["SPY"].iloc[-1]

# Simulate modest growth (annualized return assumptions)
bnd_growth_rate = 0.02   # 2% annual
spy_growth_rate = 0.10   # 10% annual

forecast_dates = tsla_forecast.index
bnd_forecast = pd.Series(
    [last_bnd_price * (1 + bnd_growth_rate/250) ** i for i in range(1, len(forecast_dates)+1)],
    index=forecast_dates, name="BND"
)
spy_forecast = pd.Series(
    [last_spy_price * (1 + spy_growth_rate/250) ** i for i in range(1, len(forecast_dates)+1)],
    index=forecast_dates, name="SPY"
)

# Merge into forward-looking price DataFrame
fwd_prices = pd.concat([tsla_forecast["TSLA"], bnd_forecast, spy_forecast], axis=1)

# --- 4. Calculate expected returns & covariance ---
returns = fwd_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna()

mu = expected_returns.mean_historical_return(fwd_prices)  # annualized from forecast horizon
S = risk_models.CovarianceShrinkage(returns).ledoit_wolf()

print("Expected Annual Returns (Forward-Looking):")
print(mu)

# --- 5. Optimize portfolio ---
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()
cleaned_weights = ef.clean_weights()

print("\nOptimized Portfolio Weights:")
for asset, weight in cleaned_weights.items():
    print(f"{asset}: {weight:.4f}")

# Portfolio performance
ef.portfolio_performance(verbose=True)

# --- 6. Save results ---
fwd_prices.to_csv("outputs/forward_prices.csv")
pd.Series(cleaned_weights, name="Weight").to_csv("outputs/forward_optimized_weights.csv")

print("\n✅ Forward-looking optimization complete. Results saved.")
