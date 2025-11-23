#!/usr/bin/env python3
"""
From Numpy
"""
import pandas as pd


def from_numpy(array):
    """
    Converts a numpy array to a pandas DataFrame.
    Args:
        array (numpy.ndarray): The input numpy array.
    Returns:
        pandas.DataFrame: The converted DataFrame.
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]

    df = pd.DataFrame(array, columns=columns)

    return df
