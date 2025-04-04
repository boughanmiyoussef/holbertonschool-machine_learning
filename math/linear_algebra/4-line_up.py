#!/usr/bin/env python3
"""
Adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """
    Adds two arrays element-wise.

    Args:
        arr1 (list): The first array (list of ints/floats).
        arr2 (list): The second array (list of ints/floats).

    Returns:
        list: A new list containing the element-wise sums.
              Returns None if the arrays are not the same length.
    """
    if len(arr1) != len(arr2):
        return None

    result = []

    for a, b in zip(arr1, arr2):
        result.append(a + b)

    return result
