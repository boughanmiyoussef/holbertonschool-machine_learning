#!/usr/bin/env python3
"""
Fill
"""


def fill(df):
    """
    Takes a pd.DataFrame and:
        Removes the Weighted_Price column.
        Fills missing values in the Close column with the previous row value
        Fills missing values in the High, Low, and Open columns with
            the corresponding Close value in the same row.
        Sets missing values in Volume_(BTC) and Volume_(Currency) to 0.
    Returns:
        the modified pd.DataFrame
    """
    df = df.drop(columns=['Weighted_Price'])

    df['Close'] = df['Close'].fillna(method='ffill')

    df['High'] = df['High'].fillna(df['Close'])

    df['Low'] = df['Low'].fillna(df['Close'])

    df['Open'] = df['Open'].fillna(df['Close'])

    df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)

    df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

    return df
