#!/usr/bin/env python3
"""
Slice
"""


def slice(df):
    """
    Takes a pd.DataFrame and:
        Extracts the columns High, Low, Close, and Volume_BTC.
        Selects every 60th row from these columns.
    Returns:
        the sliced pd.DataFrame
    """
    df1 = df[['High', 'Low', 'Close', 'Volume_(BTC)']]

    df2 = df1.iloc[::60, :]

    return df2
