#!/usr/bin/env python3
"""
Prune
"""


def prune(df):
    """
    Takes a pd.DataFrame and:
        Removes any entries where Close has NaN values
    Returns:
        the modified pd.DataFrame.
    """
    df = df.dropna()

    return df
