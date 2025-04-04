#!/usr/bin/env python3

"""
This module provides a function to calculate the shape of a matrix.
The shape is returned as a list of integers representing the dimensions.
"""


def matrix_shape(matrix):
    """
    Return the sape of matrix.
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else []
    return shape
