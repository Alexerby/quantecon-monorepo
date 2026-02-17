import numpy as np
import matplotlib.pyplot as plt

from quantecon_lib.basis_models.splines import CubicSpline, LinearSpline
from quantecon_lib.viz.core import plot_spline_fit

X = np.linspace(0, 10, 100).reshape(-1, 1)
y = np.sin(X).reshape(-1, 1)

linear_spline = LinearSpline(n_knots=3).fit(X, y)
y_pred = linear_spline.predict(X)

cubic_spline = CubicSpline(n_knots=3).fit(X, y)
y_pred_cubic = cubic_spline.predict(X)

c_knots = cubic_spline.knots
c_knots_preds = cubic_spline.predict(c_knots)
print(c_knots)


plot_spline_fit(X, y, linear_spline)
plot_spline_fit(X, y, cubic_spline)
plt.show()
