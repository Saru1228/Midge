"""
Shared helpers for particle-level PDE fitting.
"""

import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_gaussian_laplacian(pos, vec, length_scale=15.0, k_neighbors=10):
    """
    Particle-level Laplacian using KNN + Gaussian weights:
        Δ vec_i ≈ Σ_j w_ij (vec_j - vec_i) / ℓ^2
    """
    pos = np.asarray(pos)
    vec = np.asarray(vec)

    n_points = len(pos)
    n_nei = min(k_neighbors + 1, n_points)
    if n_points == 0 or n_nei <= 1:
        return np.zeros_like(vec)

    nbrs = NearestNeighbors(n_neighbors=n_nei).fit(pos)
    distances, indices = nbrs.kneighbors(pos)

    lap = np.zeros_like(vec)

    for i in range(n_points):
        idx = indices[i, 1:]      # skip self
        r2 = distances[i, 1:]**2
        if len(idx) == 0:
            continue

        w = np.exp(-r2 / (2.0 * length_scale**2))
        s = w.sum()
        if s <= 0:
            continue
        w /= s

        lap[i] = (w[:, None] * (vec[idx] - vec[i])).sum(axis=0) / (length_scale**2)

    return lap
