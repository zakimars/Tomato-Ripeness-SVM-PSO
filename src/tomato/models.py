"""Model definitions: K-Means baseline and SVM kernels with their PSO search spaces."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans
from sklearn.svm import SVC

# Search ranges from thesis Table 3.2.
C_RANGE = (0.01, 1000.0)
GAMMA_RANGE = (0.0001, 10.0)
COEF0_RANGE = (-1.0, 1.0)  # "r"
DEGREE_RANGE = (1.0, 5.0)  # "d", rounded to an integer

# For each kernel: parameter names and bounds searched by PSO.
SEARCH_SPACES: dict[str, list[tuple[str, tuple[float, float]]]] = {
    "rbf": [("C", C_RANGE), ("gamma", GAMMA_RANGE)],
    "sigmoid": [("C", C_RANGE), ("gamma", GAMMA_RANGE), ("coef0", COEF0_RANGE)],
    "poly": [("C", C_RANGE), ("gamma", GAMMA_RANGE), ("coef0", COEF0_RANGE), ("degree", DEGREE_RANGE)],
}


def decode(kernel: str, position: np.ndarray) -> dict:
    params = {name: float(v) for (name, _), v in zip(SEARCH_SPACES[kernel], position)}
    if "degree" in params:
        params["degree"] = int(round(params["degree"]))
    return params


def make_svm(kernel: str, **params) -> SVC:
    return SVC(kernel=kernel, **params)


class KMeansClassifier:
    """K-Means (k=3, k-means++, 10 inits, 300 iters) with clusters mapped to the
    majority training label, so it can be scored like a classifier."""

    def __init__(self, n_clusters: int = 3, seed: int = 42):
        self.km = KMeans(n_clusters=n_clusters, init="k-means++", n_init=10, max_iter=300, random_state=seed)
        self.mapping: dict[int, int] = {}

    def fit(self, X, y):
        clusters = self.km.fit_predict(X)
        for k in np.unique(clusters):
            self.mapping[int(k)] = int(np.bincount(y[clusters == k]).argmax())
        return self

    def predict(self, X):
        return np.array([self.mapping[int(k)] for k in self.km.predict(X)])
