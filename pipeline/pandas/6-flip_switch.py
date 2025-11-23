#!/usr/bin/env python3
"""
Flip it and Switch it
"""


def flip_switch(df):
    """
    Takes a pd.DataFrame and:
        Sorts the data in reverse chronological order.
        Transposes the sorted dataframe.
    Returns:
        the transformed pd.DataFrame.
    """
    df1 = df.sort_values(by='Timestamp', ascending=False)

    df2 = df1.transpose()

    return df2
