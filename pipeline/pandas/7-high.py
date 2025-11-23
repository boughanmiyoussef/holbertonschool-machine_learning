#!/usr/bin/env python3
"""
Sort
"""


def high(df):
    """
    Takes a pd.DataFrame and:
        Sorts it by the High price in descending order.
    Returns:
        the sorted pd.DataFrame
    """
    df = df.sort_values(by='High', ascending=False)

    return df
