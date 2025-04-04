#!/usr/bin/env python3
"""
This module provides a function to transpose a 2D matrix.
The transpose of a matrix is obtained by flipping the matrix over its diagonal,
which means rows become columns and columns become rows.
"""


def matrix_transpose(matrix):
    """
    Returns the transpose of a 2D matrix.
    """
    return list(map(list, zip(*matrix)))
