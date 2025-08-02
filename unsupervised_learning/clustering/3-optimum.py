#!/usr/bin/env python3
"""Contains optimum function."""


import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """tests for optimum number of clusters by variance."""
    results = []
    d_vars = []

    baseline_var = None
    for k in range(kmin, kmax + 1):
        c, clss = kmeans(X, k, iterations)
        var = variance(X, c)
        results.append((c, clss))
        d_vars.append(abs(
            var - (baseline_var if baseline_var is not None else(
                baseline_var := var
            ))
        ))

    return results, d_vars