import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# 1. Download historical data
ticker = "AAPL"
data = yf.download(ticker, start="2023-01-01", end="2024-01-01")

prices = data["Adj Close"]

# 2. Compute log returns & volatility
log_returns = np.log(prices / prices.shift(1)).dropna()

sigma = log_returns.std() * np.sqrt(252)

print(f"Estimated annual volatility: {sigma:.2%}")

# 3. Option & model parameters
S0 = prices.iloc[-1]   # current stock price
K = round(S0)          # ATM option
r = 0.05               # risk-free rate
T = 1.0                # 1 year maturity

N = 252                # time steps
M = 100_000            # simulations
dt = T / N

# 4. Monte Carlo simulation (GBM)
Z = np.random.standard_normal((M, N))

S = np.zeros((M, N + 1))
S[:, 0] = S0

for t in range(1, N + 1):
    S[:, t] = S[:, t-1] * np.exp(
        (r - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * Z[:, t-1]
    )

# 5. Option payoff
ST = S[:, -1]
payoffs = np.maximum(ST - K, 0)

# 6. Discounted price & confidence interval
discount_factor = np.exp(-r * T)
price_mc = discount_factor * np.mean(payoffs)

std_error = discount_factor * np.std(payoffs) / np.sqrt(M)
ci_lower = price_mc - 1.96 * std_error
ci_upper = price_mc + 1.96 * std_error

print(f"\nMonte Carlo Call Price: {price_mc:.2f}")
print(f"95% CI: [{ci_lower:.2f}, {ci_upper:.2f}]")

# 7. Visualization
plt.figure(figsize=(10, 6))
for i in range(20):
    plt.plot(S[i])
plt.title(f"Monte Carlo Price Paths for {ticker}")
plt.xlabel("Time Steps")
plt.ylabel("Stock Price")
plt.show()

plt.hist(ST, bins=100, density=True)
plt.title("Distribution of Final Stock Prices")
plt.xlabel("Final Price")
plt.show()
