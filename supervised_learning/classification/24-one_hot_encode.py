#!/usr/bin/env python3
"""
Module for one-hot encoding of numeric labels.
"""
import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if classes <= 0 or Y.ndim != 1 or np.max(Y) >= classes:
        return None

    m = Y.shape[0]
    one_hot = np.zeros((classes, m))
    rows = Y.astype(int)
    one_hot[rows, np.arange(m)] = 1
    return one_hot
