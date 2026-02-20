import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from quantecon_lib.ensemble import (
    BaggingRegressor,
    GradientBoostingRegressor,
    AdaBoostRegressor,
)
from quantecon_lib.nn import FeedForwardNetwork
from quantecon_lib.nn.activations import ReLU, Sigmoid

from quantecon_lib.nn.layers import Dense
from quantecon_lib.nn.losses import MSE
from quantecon_lib.nn.optimizers import SGD
from quantecon_lib.viz import use_quantecon_style

use_quantecon_style()

# Data
np.random.seed(42)
x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)
y = np.sin(x).ravel() + np.random.normal(0, 0.3, 100)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

# Bagging
bagging = BaggingRegressor(n_estimators=1000, max_depth=3)
bagging.fit(x_train, y_train)
bagging_line = bagging.predict(x)

# Gradient Boosting
gbr = GradientBoostingRegressor()
gbr.fit(x_train, y_train, eval_set=(x_test, y_test), tol=1e-6, patience=100)
gbr_preds = gbr.predict(x)

# AdaBoost
ada = AdaBoostRegressor(n_estimators=100, max_depth=3)
ada.fit(x_train, y_train)
ada_preds = ada.predict(x)

# FeedForwardNetwork
nn_layers = [
    Dense(1, 64),    ReLU(),
    Dense(64, 32),   ReLU(),
    Dense(32, 1)     # No activation on the last layer for regression
]


# Plotting
plt.figure(figsize=(10, 6))
plt.title("Bagging vs Boosting: Sine Wave Reconstruction")

plt.scatter(x_train, y_train, color="black", s=15, alpha=0.3, label="Train Data")
plt.plot(x, bagging_line, label="Bagging (Variance Reduction)")
plt.plot(x, gbr_preds, label="Gradient Boosting")
plt.plot(x, ada_preds, label="Ada Boosting")
plt.plot(x, np.sin(x), color="gray", linestyle="--", alpha=0.5, label="True Sine")

plt.legend()
plt.show()
