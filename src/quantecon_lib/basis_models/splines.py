import numpy as np
from ._base import BaseBasisModel


class BaseSplineModel(BaseBasisModel):
    def __init__(self, n_knots=3):
        super().__init__()
        self.n_knots = n_knots
        self.knots = None


    def _get_knots(self, X):
        if self.knots is None:
            quantiles = np.linspace(0, 100, self.n_knots + 2)[1:-1]
            self.knots = np.percentile(X, quantiles)
        return self.knots

class LinearSpline(BaseSplineModel):
    def __init__(self, n_knots=3):
        super().__init__(n_knots=n_knots)

    def _transform(self, X):
        if self.knots is None:
            self.knots = self._get_knots(X)

        n_samples = X.shape[0]
        H = np.hstack([np.ones((n_samples, 1)), X])

        for knot in self.knots:
            hinge = np.maximum(0, X - knot)
            H = np.hstack([H, hinge])

        return H

class CubicSpline(BaseSplineModel):
    def __init__(self, n_knots=3):
        super().__init__(n_knots=n_knots)

    def _transform(self, X):
        if self.knots is None:
            self.knots = self._get_knots(X)

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


class SmoothingSpline(BaseSplineModel):
    def __init__(self, lam=1.0):
        super().__init__(n_knots=-1)
        self.lam = lam

    def _compute_penalty_matrix(self,knots):
        n = len(knots)
        h = np.diff(knots)

        Q = np.zeros((n, n-2))

        for j in range(n-2):
            Q[j, j] = 1.0 / h[j]
            Q[j+1, j] = -(1.0 / h[j] + 1.0 / h[j+1])
            Q[j+2, j] = 1.0 / h[j+1]

        R = np.zeros((n-2, n-2))
        for j in range(n-2):
            R[j, j] = (h[j] + h[j+1]) / 3.0
            if j < n-3:
                R[j, j+1] = R[j+1, j] = h[j+1] / 6.0

        return Q @ np.linalg.solve(R, Q.T)


    def _transform(self, X):

        # If we are fitting (X same as knots)
        if self.knots is not None and np.array_equal(X, self.knots):
            return np.eye(len(X))
        
        return self._build_natural_basis(X, self.knots)

    def _build_natural_basis(self, X, knots):
        n_k = len(knots)
        n_s = len(X)
        X_flat = X.flatten()

        # Pre-allocate H
        H = np.zeros((n_s, n_k))
        H[:, 0] = 1
        H[:, 1] = X_flat

        def d(k_idx, x_val):
            # The numerator logic ensures cubic terms cancel at boundaries
            num = np.maximum(0, x_val - knots[k_idx])**3 - np.maximum(0, x_val - knots[-1])**3
            den = knots[-1] - knots[k_idx]
            return num / den

        for j in range(n_k - 2):
            # This logic enforces the "Natural" boundary constraint
            H[:, j + 2] = d(j, X_flat) - d(n_k - 2, X_flat)

        return H

    def _get_knots(self, X):
        return np.unique(X)

    def fit(self, X, y):
        self.knots = self._get_knots(X)
        H = self._transform(X)
        
        # Compute the penalty matrix Omega
        self.omega = self._compute_penalty_matrix(self.knots)
        
        # Solve (H.T @ H + lam * Omega) * beta = H.T @ y
        A = H.T @ H + self.lam * self.omega
        b = H.T @ y
        self.beta = np.linalg.solve(A, b)

        return self

    def predict(self, X):
            H = self._transform(X)
            return H @ self.beta
