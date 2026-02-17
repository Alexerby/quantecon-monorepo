import numpy as np
import matplotlib.pyplot as plt
from quantecon_lib.ensemble.bagging import BaggingRegressor

np.random.seed(42)
X = np.sort(5 * np.random.rand(200, 1), axis=0)
y = np.sin(X).ravel()
y += 0.15 * np.random.normal(size=y.shape)

model = BaggingRegressor(n_estimators=50, max_depth=5)
model.fit(X, y)

X_test = np.linspace(0, 5, 500)[:, np.newaxis]
y_hat = model.predict(X_test)

plt.figure(figsize=(10, 6))
plt.scatter(X, y, color='gray', alpha=0.5, label='Noisy Data')
plt.plot(X_test, np.sin(X_test), color='blue', label='True Sine', linewidth=2)
plt.plot(X_test, y_hat, color='red', label='Prediction', linewidth=2)
plt.title(f"Regression Fit (R²: {model.r2:.3f})")
plt.legend()
plt.show()
