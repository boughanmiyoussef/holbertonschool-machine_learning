#!/usr/bin/env python3


def add_matrices2D(mat1, mat2):
    """
    """

    if len(mat1) != len(mat2):
        return None

    if len(mat1[0]) != len(mat2[0]):
        return None

    result = []

    for row1, row2 in zip(mat1, mat2):
        result_row = [x + y for x, y in zip(row1, row2)]
        result.append(result_row)

    return result
