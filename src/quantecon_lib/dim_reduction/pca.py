import numpy as np


class PCA:
    def __init__(self, X: np.ndarray, k: int = 1) -> None:
        self.X = X
        self.k = k
        self.mean = None
        self.explained_variance_ = None

        self.components: np.ndarray = np.empty((0, 0))
        self.cov_matrix: np.ndarray = np.empty((0, 0))

    def _standardize_data(self) -> np.ndarray:
        for col in range(self.X.shape[1]):
            col_data = self.X[:, col]
            mean = np.mean(col_data)
            sigma = np.std(col_data)
            self.X[:, col] = (col_data - mean) / (sigma + 1e-8)
        return self.X

    def _calculate_covariance_matrix(self) -> np.ndarray:
        return np.cov(self.X, rowvar=False)

    def fit(self):
        self._standardize_data()
        cov_matrix = self._calculate_covariance_matrix()

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Store results
        self.components = eigenvectors[:, : self.k]
        self.explained_variance_ = eigenvalues[: self.k]

        return self

    def transform(self) -> np.ndarray:
        return np.dot(self.X, self.components)
