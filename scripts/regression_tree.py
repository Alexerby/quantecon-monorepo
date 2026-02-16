import numpy as np
import pandas as pd

from quantecon_lib.ensemble.classification_trees import build_tree


np.random.seed(42)
X = pd.DataFrame({'feature_1': np.linspace(-5, 5, 20)})
y = X['feature_1']**2 + np.random.normal(0, 2, 20)

t = build_tree(X, y, 5, X.columns)
print(t)
