"""PCA, kNN and k-means written the way they are done in the FOML lab notebooks (step by step in NumPy)."""
import numpy as np


class LabPCA:
    """Same steps as the lab PCA notebook: mean -> centre -> covariance -> eigen -> sort -> project."""

    def __init__(self, n_components):
        self.n_components = n_components

    def fit(self, a):
        a = np.asarray(a, dtype=np.float64)
        self.mean_ = np.mean(a.T, axis=1)                   # m = np.mean(a.T, axis=1)
        c = a - self.mean_                                  # c = a - m
        V = np.cov(c.T)                                     # covariance matrix
        eigenvalues, eigenvector = np.linalg.eigh(V)        # eigh = eig for a symmetric matrix (faster, real numbers)
        idx = eigenvalues.argsort()[::-1]                   # biggest eigenvalue first
        self.eigenvalues_ = eigenvalues[idx]
        self.eigenvectors_ = eigenvector[:, idx]
        self.explained_variance_ratio_ = self.eigenvalues_ / self.eigenvalues_.sum()
        self.components_ = self.eigenvectors_[:, : self.n_components].T
        return self

    def transform(self, a, n=None):
        n = n or self.n_components
        return np.dot(np.asarray(a, dtype=np.float64) - self.mean_, self.eigenvectors_[:, :n])   # Z = c . eig


class LabKNN:
    """k nearest neighbours by cosine similarity (rows are unit length, so cosine = dot product)."""

    def __init__(self, Z):
        self.Z = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12)

    def similarity(self, z):
        z = z / (np.linalg.norm(z) + 1e-12)
        return self.Z @ z

    def query(self, z, k=10, exclude=None):
        s = self.similarity(z)
        if exclude is not None:
            s[exclude] = -2
        top = np.argpartition(-s, k)[:k]
        top = top[np.argsort(-s[top])]
        return top, s[top]


class LabKMeans:
    """k-means: pick k starting centres, assign each point to the nearest centre, move centres to the mean, repeat."""

    def __init__(self, k, iters=50, seed=0):
        self.k, self.iters, self.seed = k, iters, seed

    def fit(self, X):
        rng = np.random.default_rng(self.seed)
        n = len(X)
        centres = [X[rng.integers(n)]]                      # k-means++ start: spread the first centres out
        for _ in range(1, self.k):
            d2 = np.min(((X[:, None, :] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
            centres.append(X[rng.choice(n, p=d2 / d2.sum())])
        C = np.array(centres)
        for _ in range(self.iters):
            d = ((X ** 2).sum(1)[:, None] - 2 * X @ C.T + (C ** 2).sum(1)[None])
            lab = d.argmin(1)
            newC = np.array([X[lab == j].mean(0) if (lab == j).any() else C[j] for j in range(self.k)])
            if np.allclose(newC, C):
                break
            C = newC
        self.centres_, self.labels_ = C, lab
        self.inertia_ = float(((X - C[lab]) ** 2).sum())
        return self

    def predict(self, X):
        d = ((X ** 2).sum(1)[:, None] - 2 * X @ self.centres_.T + (self.centres_ ** 2).sum(1)[None])
        return d.argmin(1)
