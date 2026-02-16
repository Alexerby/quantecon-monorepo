import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from quantecon_lib.ensemble.bagging import bagging_regressor
from quantecon_lib.ensemble.boosting import gradient_boosting
from quantecon_lib.ensemble.classification_trees import DecisionTree



# --------------------------------------
# DATA
# --------------------------------------
np.random.seed(42)
n_samples = 200 

X = pd.DataFrame({
    "x": np.linspace(0, 10, 200)
    })

# y = sin(x) + \epsilon, \epsilon ~ N(0, 0.15)
y = np.sin(X['x']) + np.random.normal(0, 0.15, n_samples) 


# --------------------------------------
# METHODS
# --------------------------------------
predictions = gradient_boosting(
        y=y,
        X=X,
        M=100,
        learning_rate=0.1
)

# --------------------------------------

dt3 = DecisionTree(max_depth=3)
dt3.fit(X ,y)
dt3_predictions =dt3.predict(X)


# --------------------------------------

br = bagging_regressor(
        y=y,
        X=X,
        max_depth=3
)

# --------------------------------------


plt.scatter(X["x"], y)
plt.plot(X["x"], predictions, color="red", label="Gradient Boosting Prediction")
plt.plot(X["x"], dt3_predictions, color="green", label="Individual tree (max_depth=3)")
plt.plot(X["x"], br, color="blue", label="Bagging Regressor")


plt.legend()
plt.show()
