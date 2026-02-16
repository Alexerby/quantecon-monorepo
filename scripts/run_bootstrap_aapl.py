import numpy as np

from quantecon_lib.stats.bootstrapping import bootstrap


# --- DATA PREPARATION ---
# Prices: Mar 2025 to Feb 2026
prices = np.array([
    221.17, 211.58, 199.98, 204.55, 206.94, 231.44, 
    254.15, 269.86, 278.32, 271.61, 259.24, 255.54
])

# Calculate 11 monthly returns
aapl_returns = np.diff(prices) / prices[:-1]

# --- ANALYSIS ---
# Set seed for reproducibility
np.random.seed(42)

# 1. Parametric SE Calculation
n = len(aapl_returns)
mu_hat = np.mean(aapl_returns)
# Sample standard deviation (s)
s = np.std(aapl_returns, ddof=1)
# Formula: SE = s / sqrt(n)
se_parametric = s / np.sqrt(n)

# 2. Bootstrap SE Calculation
# Passing np.mean as our estimator function
boot_mean, boot_se, boot_dist = bootstrap(aapl_returns, estimator=np.mean, B=5000)

# --- RESULTS ---
print("-" * 30)
print(f"Sample size (n): {n}")
print(f"Mean Estimate:   {mu_hat:.4%}")
print("-" * 30)
print(f"Parametric SE:   {se_parametric:.4f}")
print(f"Bootstrap SE:    {boot_se:.4f}")
print("-" * 30)

# Calculate t-stats (H0: mu = 0)
t_parametric = mu_hat / se_parametric
t_boot = mu_hat / boot_se

print(f"t-stat (Parametric): {t_parametric:.2f}")
print(f"t-stat (Bootstrap):  {t_boot:.2f}")
