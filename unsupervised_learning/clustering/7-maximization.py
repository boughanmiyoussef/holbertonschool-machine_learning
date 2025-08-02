
#!/usr/bin/env python3
# 7-maximization.py
"""7-maximization.py"""


import numpy as np


def maximization(X, g):
    """maximization"""
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or g.ndim != 2:
        return None, None, None
    if g.shape[1] != X.shape[0]:
        return None, None, None
    if not np.allclose(g.sum(axis=0), 1.0):
        return None, None, None

    k, n = g.shape

    # Calcul des priorités pi
    pi = np.sum(g, axis=1) / np.sum(g)

    # Calcul des centres m
    m = np.dot(g, X) / np.sum(g, axis=1, keepdims=True)

    # Calcul des matrices de covariance S avec une boucle
    S = np.zeros((k, X.shape[1], X.shape[1]))  # Initialisation de S

    for i in range(k):
        X_centered = X - m[i]  # Centrer X par rapport à m[i]
        S[i] = np.dot((g[i] * X_centered.T), X_centered) / np.sum(g[i])

    return pi, m, S
