import numpy as np
from sklearn.datasets import make_moons
from quantecon_lib.ensemble.forest import RandomForestClassifier
from quantecon_lib.ensemble.boosting import GradientBoostingRegressor
from quantecon_lib.ensemble.bagging import BaggingRegressor

# Test A: Random Forest Classifier
X_c, y_c = make_moons(n_samples=100, noise=0.1)
rf = RandomForestClassifier(n_estimators=10, max_depth=3)
rf.fit(X_c, y_c)
rf_acc = np.mean(rf.predict(X_c) == y_c)
print(f"RF Classifier Test: {rf_acc * 100:.1f}% Accuracy")

# Test B: Gradient Boosting Regressor
X_r = np.linspace(0, 5, 100).reshape(-1, 1)
y_r = np.sin(X_r).ravel()
gbr = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1)
gbr.fit(X_r, y_r)
print(f"GBR Regressor Test: Fitted {len(gbr.models)} trees")

# Test C: Bagging Regressor
X_b = np.linspace(-3, 3, 100).reshape(-1, 1)
y_b = X_b.ravel()**2 + np.random.normal(0, 0.5, 100)
br = BaggingRegressor(n_estimators=20, max_depth=4).fit(X_b, y_b)
print(f"Bagging Regressor Test: R² = {br.r2:.3f}")
