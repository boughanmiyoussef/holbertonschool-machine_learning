#!/usr/bin/env python3
"""
To Numpy
"""


def array(df):
    """
    Takes a pd.DataFrame as input and performs the following:
        df is a pd.DataFrame containing columns named High and Close
        should select the last 10 rows of the High and Close columns.
        Convert these selected values into a numpy.ndarray.
    Returns:
        the numpy.ndarray
    """
    df1 = df.iloc[-10:][['High', 'Close']]

    # or we can convert by arr = df1.values
    return df1.to_numpy()
