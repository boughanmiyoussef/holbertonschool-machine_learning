#!/usr/bin/env python3
"""
Clustering module
"""
import numpy as np


def pdf(X, m, S):
    """
    Calculates the probability density function of a Gaussian distribution
    Args:
        X is a numpy.ndarray of shape (n, d)
        m is a numpy.ndarray of shape (d,)
        S is a numpy.ndarray of shape (d, d)
    Returns:
        P, or None
            P is a numpy.ndarray of shape (n,)
    All values in P should have a minimum value of 1e-300
    """
    if (not isinstance(X, np.ndarray) or not isinstance(m, np.ndarray)
            or not isinstance(S, np.ndarray)):
        return None

    if len(X.shape) != 2 or len(m.shape) != 1 or len(S.shape) != 2:
        return None

    n, d = X.shape

    if m.shape[0] != d or S.shape != (d, d):
        return None

    try:
        det = np.linalg.det(S)

        inv = np.linalg.inv(S)

        norm_const = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)

        diff = X - m

        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)

        P = norm_const * np.exp(exponent)

        return np.maximum(P, 1e-300)
    except Exception:
        return None
