"""
File: task5_backtesting.py
Author: Financial Analyst - GMF Investments
Date: August 2025
Description: Backtest the optimized portfolio strategy vs. 60/40 benchmark.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load Data
# -----------------------------
print("Loading processed price data...")
prices = pd.read_csv("Data/processed_prices.csv", index_col=0, parse_dates=[0])
prices.index = pd.to_datetime(prices.index, utc=True).tz_localize(None)

# Define backtest period: last 12 months
backtest_start = '2024-08-01'
backtest_end = '2025-07-31'
backtest_prices = prices[backtest_start:backtest_end]

# Ensure we have data
print(f"Backtesting from {backtest_start} to {backtest_end} → {len(backtest_prices)} trading days")

# -----------------------------
# 2. Portfolio Weights
# -----------------------------
# Use weights from Task 4 (Max Sharpe Portfolio)
weights_optimal = {
    'TSLA': 0.582,
    'BND':  0.186,
    'SPY':  0.232
}

# Benchmark: 60% SPY / 40% BND
weights_benchmark = {
    'TSLA': 0.000,
    'BND':  0.400,
    'SPY':  0.600
}

# -----------------------------
# 3. Calculate Daily Returns
# -----------------------------
returns = backtest_prices.pct_change().dropna()

# -----------------------------
# 4. Simulate Portfolio Returns
# -----------------------------
# Optimal strategy returns
return_optimal = (returns * pd.Series(weights_optimal)).sum(axis=1)

# Benchmark returns
return_benchmark = (returns * pd.Series(weights_benchmark)).sum(axis=1)

# Cumulative returns (growth of $1)
cumulative_optimal = (1 + return_optimal).cumprod()
cumulative_benchmark = (1 + return_benchmark).cumprod()

# -----------------------------
# 5. Performance Metrics
# -----------------------------
def portfolio_metrics(returns, name):
    total_return = (returns + 1).prod() - 1
    annualized_return = (1 + total_return) ** (252/len(returns)) - 1
    annualized_vol = returns.std() * np.sqrt(252)
    sharpe_ratio = annualized_return / annualized_vol if annualized_vol > 0 else 0
    return {
        "Portfolio": name,
        "Total Return": total_return,
        "Annual Return": annualized_return,
        "Annual Volatility": annualized_vol,
        "Sharpe Ratio": sharpe_ratio
    }

metrics_optimal = portfolio_metrics(return_optimal, "Optimal Portfolio")
metrics_benchmark = portfolio_metrics(return_benchmark, "60/40 Benchmark")

results = pd.DataFrame([metrics_optimal, metrics_benchmark]).set_index("Portfolio")
print("\n📊 Backtesting Results:")
print(results.round(4))

# Save results
results.to_csv("outputs/backtest_results.csv")

# -----------------------------
# 6. Plot Cumulative Returns
# -----------------------------
plt.figure(figsize=(14, 7))
plt.plot(cumulative_optimal.index, cumulative_optimal, label='Optimal Portfolio (TSLA-Optimized)', color='blue')
plt.plot(cumulative_benchmark.index, cumulative_benchmark, label='60/40 Benchmark (SPY/BND)', color='orange')
plt.title("Backtest: Optimal Portfolio vs 60/40 Benchmark (Aug 2024 – Jul 2025)")
plt.xlabel("Date")
plt.ylabel("Growth of $1")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("outputs/backtest_performance.png", dpi=300)
plt.show()

# -----------------------------
# 7. Final Summary
# -----------------------------
optimal_final = cumulative_optimal.iloc[-1]
benchmark_final = cumulative_benchmark.iloc[-1]

print(f"\n📈 Final Value of $1:")
print(f"  Optimal Portfolio: ${optimal_final:.2f}")
print(f"  60/40 Benchmark:   ${benchmark_final:.2f}")

if optimal_final > benchmark_final:
    print("\n✅ CONCLUSION: Your model-driven strategy outperformed the benchmark.")
    print("This supports the use of forecasting and optimization in portfolio management.")
else:
    print("\n⚠️  CONCLUSION: The benchmark outperformed your strategy.")
    print("Consider refining the forecast or rebalancing logic.")

print("\n✅ Backtesting complete. Results saved to 'outputs/' folder.")