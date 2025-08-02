#!/usr/bin/env python3
# 6-expectation.py
"""6-expectation"""


import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """Calculates the expectation step in the EM algorithm for a GMM"""
    if (not isinstance(X, np.ndarray) or X.ndim != 2 or
            not isinstance(pi, np.ndarray) or pi.ndim != 1 or
            not isinstance(m, np.ndarray) or m.ndim != 2 or
            not isinstance(S, np.ndarray) or S.ndim != 3 or
            X.shape[1] != m.shape[1] or m.shape[0] != S.shape[0] or
            S.shape[1] != S.shape[2] or
            pi.shape[0] != m.shape[0] or pi.shape[0] != S.shape[0]):
        return None, None

    # Vérification de la validité de pi
    if not np.isclose(np.sum(pi), 1) or np.any(pi < 0) or np.any(pi > 1):
        return None, None

    k, n = pi.shape[0], X.shape[0]
    g = np.zeros((k, n))

    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    total_prob = np.sum(g, axis=0)
    g /= total_prob

    total_prob = np.maximum(total_prob, 1e-300)
    like = np.sum(np.log(total_prob))

    return g, like
