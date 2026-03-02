# IMPORTS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.stats import norm

def monte_carlo_call_price(S0, K, r, sigma, T, N, M):
    dt = T / N

    Z = np.random.standard_normal((M, N))

    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    for t in range(1, N + 1):
        S[:, t] = S[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z[:, t-1]
        )

    ST = S[:, -1]
    payoffs = np.maximum(ST - K, 0)

    return np.exp(-r * T) * np.mean(payoffs)

def monte_carlo_call_price_with_Z(S0, K, r, sigma, T, N, Z):
    dt = T / N
    M = Z.shape[0]

    S = np.zeros((M, N + 1))
    S[:, 0] = S0

    for t in range(1, N + 1):
        S[:, t] = S[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt
            + sigma * np.sqrt(dt) * Z[:, t-1]
        )

    ST = S[:, -1]
    payoffs = np.maximum(ST - K, 0)

    return np.exp(-r * T) * np.mean(payoffs)

def black_scholes_delta_call(S0, K, r, sigma, T):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

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

# 7. STABLE DELTA & GAMMA (COMMON RANDOM NUMBERS)

M_greeks = 200_000
bump = 0.05

# ONE shared random matrix
Z_common = np.random.standard_normal((M_greeks, N))

price_down = monte_carlo_call_price_with_Z(
    S0 - bump, K, r, sigma.item(), T, N, Z_common
)

price_mid = monte_carlo_call_price_with_Z(
    S0, K, r, sigma.item(), T, N, Z_common
)

price_up = monte_carlo_call_price_with_Z(
    S0 + bump, K, r, sigma.item(), T, N, Z_common
)

delta_mc = (price_up - price_mid) / bump

delta_down = (price_mid - price_down) / bump
delta_up = (price_up - price_mid) / bump

gamma_mc = (delta_up - delta_down) / bump

print("\n--- STABLE GREEKS ---")
print(f"Monte Carlo Delta: {delta_mc:.4f}")
print(f"Monte Carlo Gamma: {gamma_mc:.6f}")

# 8. DELTA & GAMMA VS STOCK PRICE (VISUALIZATION)

price_range = np.linspace(0.8 * S0, 1.2 * S0, 25)
bump = 0.05
M_plot = 100_000

Z_common = np.random.standard_normal((M_plot, N))

delta_values = []
gamma_values = []

for S_test in price_range:

    price_down = monte_carlo_call_price_with_Z(
        S_test - bump, K, r, sigma.item(), T, N, Z_common
    )

    price_mid = monte_carlo_call_price_with_Z(
        S_test, K, r, sigma.item(), T, N, Z_common
    )

    price_up = monte_carlo_call_price_with_Z(
        S_test + bump, K, r, sigma.item(), T, N, Z_common
    )

    delta = (price_up - price_mid) / bump

    delta_down = (price_mid - price_down) / bump
    delta_up = (price_up - price_mid) / bump

    gamma = (delta_up - delta_down) / bump

    delta_values.append(delta)
    gamma_values.append(gamma)

# PLOT DELTA
plt.figure(figsize=(10, 5))
plt.plot(price_range, delta_values, marker="o")
plt.axvline(S0, linestyle="--", label="Current Price")
plt.title("Monte Carlo Delta vs Stock Price")
plt.xlabel("Stock Price")
plt.ylabel("Delta")
plt.legend()
plt.grid()
plt.show()

# PLOT GAMMA
plt.figure(figsize=(10, 5))
plt.plot(price_range, gamma_values, marker="o", color="orange")
plt.axvline(S0, linestyle="--", label="Current Price")
plt.title("Monte Carlo Gamma vs Stock Price")
plt.xlabel("Stock Price")
plt.ylabel("Gamma")
plt.legend()
plt.grid()
plt.show()

# 9. DELTA HEDGING SIMULATION (ONE PATH)

np.random.seed(42)

Z_path = np.random.standard_normal(N)

S_path = np.zeros(N + 1)
S_path[0] = S0

for t in range(1, N + 1):
    S_path[t] = S_path[t-1] * np.exp(
        (r - 0.5 * sigma.item()**2) * dt
        + sigma.item() * np.sqrt(dt) * Z_path[t-1]
    )

cash = 0.0
stock_position = 0.0

# One shared random matrix for Greeks
M_hedge = 50_000
Z_common = np.random.standard_normal((M_hedge, N))

deltas = []

for t in range(N):
    T_remaining = T - t * dt
    if T_remaining <= 0:
        break

    bump = 0.05

    price_down = monte_carlo_call_price_with_Z(
        S_path[t] - bump, K, r, sigma.item(), T_remaining, N, Z_common
    )

    price_mid = monte_carlo_call_price_with_Z(
        S_path[t], K, r, sigma.item(), T_remaining, N, Z_common
    )

    price_up = monte_carlo_call_price_with_Z(
        S_path[t] + bump, K, r, sigma.item(), T_remaining, N, Z_common
    )

    delta = (price_up - price_mid) / bump
    deltas.append(delta)

    # Adjust stock position
    delta_change = delta - stock_position
    cash -= delta_change * S_path[t]
    stock_position = delta

# Option payoff (short call)
option_payoff = -max(S_path[-1] - K, 0)

# Close stock position
cash += stock_position * S_path[-1]

total_pnl = cash + option_payoff

print(f"\nDelta-Hedged P&L: {total_pnl:.2f}")

# VALIDATION
bs_delta = black_scholes_delta_call(
    S0, K, r, sigma.item(), T
)

print(f"Black–Scholes Delta:        {bs_delta:.4f}")
print(f"Monte Carlo Delta:          {delta_mc:.4f}")
print(f"Difference:                {delta_mc - bs_delta:.4f}")
