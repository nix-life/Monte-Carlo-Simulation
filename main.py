# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm


# 1. DOWNLOAD REAL STOCK DATA

ticker = "NVDA"

data = yf.download(ticker, start="2023-01-01", end="2024-01-01")
prices = data["Close"]

S0 = prices.iloc[-1].item()   # Current stock price (as scalar)

# 2. ESTIMATE VOLATILITY FROM REAL DATA

log_returns = np.log(prices / prices.shift(1)).dropna()

sigma = log_returns.std() * np.sqrt(252)

print(f"Estimated annual volatility: {sigma.item():.2%}")

# 3. OPTION & MODEL PARAMETERS

K = round(S0)     # At-the-money option (now S0 is scalar, K will be int)
r = 0.05          # Risk-free rate
T = 1.0           # 1 year to maturity

N = 252           # Time steps (days)
dt = T / N

# 4. BLACK–SCHOLES (REFERENCE PRICE)

def black_scholes_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

bs_price = black_scholes_call(S0, K, r, sigma.item(), T)

print(f"Black–Scholes Call Price: {bs_price:.2f}")


# 5. MONTE CARLO CONVERGENCE TEST

simulation_sizes = [1_000, 5_000, 10_000, 50_000, 100_000]

print("\nSimulations | MC Price | Difference from BS")
print("-" * 50)

for M in simulation_sizes:

    # Generate random shocks
    Z = np.random.standard_normal((M, N))

    # Stock price paths
    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    for t in range(1, N + 1):
        S[:, t] = S[:, t-1] * np.exp(
            (r - 0.5 * sigma.item()**2) * dt
            + sigma.item() * np.sqrt(dt) * Z[:, t-1]
        )

    # Final prices
    ST = S[:, -1]

    # Option payoff
    payoffs = np.maximum(ST - K, 0) # K is already scalar here

    # Discounted Monte Carlo price
    mc_price = np.exp(-r * T) * np.mean(payoffs)

    diff = mc_price - bs_price

    print(f"{M:11d} | {mc_price:8.2f} | {diff: .4f}")


# 6. VISUALIZATION (OPTIONAL BUT IMPORTANT)

# Plot some price paths
plt.figure(figsize=(10, 6))
for i in range(20):
    plt.plot(S[i])
plt.title(f"Monte Carlo Price Paths for {ticker}")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.show()

# Distribution of final prices
plt.hist(ST, bins=100, density=True)
plt.title("Distribution of Final Stock Prices")
plt.xlabel("Final Price")
plt.show()
