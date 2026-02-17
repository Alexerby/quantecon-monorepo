import numpy as np
from ._base import BaseBasisModel, get_quantile_knots

class LinearSpline(BaseBasisModel):
    def __init__(self, n_knots=3):
        super().__init__()
        self.n_knots = n_knots

    def _transform(self, X):
        if self.knots is None:
            self.knots = get_quantile_knots(X, self.n_knots)

        n_samples = X.shape[0]
        H = np.hstack([np.ones((n_samples, 1)), X])

        for knot in self.knots:
            hinge = np.maximum(0, X - knot)
            H = np.hstack([H, hinge])

        return H

class CubicSpline(BaseBasisModel):
    def __init__(self, n_knots=3):
        super().__init__()
        self.n_knots = n_knots

    def _transform(self, X):
        if self.knots is None:
            self.knots = get_quantile_knots(X, self.n_knots)

        n_samples = X.shape[0]

        H = np.hstack([
            np.ones((n_samples, 1)),
            X,
            X**2,
            X**3
        ])

        for knot in self.knots:
            hinge = np.maximum(0, X-knot)**3
            H = np.hstack([H, hinge])

        return H

