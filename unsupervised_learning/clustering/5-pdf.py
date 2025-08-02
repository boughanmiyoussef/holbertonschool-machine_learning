#!/usr/bin/env python3
# 5-pdf.py
"""5-pdf.py"""


import numpy as np


def pdf(X, m, S):
    """
    calculates the probability density function of a Gaussian distribution
    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        return None
    if not isinstance(m, np.ndarray) or m.ndim != 1:
        return None
    if not isinstance(S, np.ndarray) or S.ndim != 2:
        return None
    if (X.shape[1] != m.shape[0] or S.shape[0] != S.shape[1]
            or S.shape[0] != m.shape[0]):
        return None

    det_cov = np.linalg.det(S)
    if det_cov <= 0:  # Check for invalid determinant
        return np.full(X.shape[0], 1e-300)

    inv_cov = np.linalg.inv(S)
    X_centered = X - m
    coef = 1 / np.sqrt((2 * np.pi) ** X.shape[1] * det_cov)
    exponent = -0.5 * np.sum(X_centered @ inv_cov * X_centered, axis=1)

    P = coef * np.exp(exponent)
    P = np.maximum(P, 1e-300)
    return P