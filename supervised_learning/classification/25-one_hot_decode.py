#!/usr/bin/env python3
"""
This module defines the one_hot_decode function for converting
a one-hot matrix back to a numeric label vector.
"""
import numpy as np


def one_hot_decode(one_hot):
    """
    Converts a one-hot matrix into a vector of labels.
    """
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    # Using np.argmax to decode the one-hot encoded matrix
    labels = np.argmax(one_hot, axis=0)
    return labels
