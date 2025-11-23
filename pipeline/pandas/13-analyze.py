#!/usr/bin/env python3
"""
Analyze
"""


def analyze(df):
    """
    Takes a pd.DataFrame and:
        Computes statistics for all columns except the Timestamp column.
    Returns:
        new pd.DataFrame containing these statistics.
    """
    df = df.drop(columns=['Timestamp'])

    df = df.describe()

    return df
