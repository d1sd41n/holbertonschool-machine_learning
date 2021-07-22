#!/usr/bin/env python3
"""[summary]

    Returns:
        [type]: [description]
    """
import pandas as pd


def from_numpy(array):
    """[summary]

    Args:
        array ([type]): [description]

    Returns:
        [type]: [description]
    """

    col = list('ABCDEFGHIJKLMNOPQRSTUVWXYZ')

    col_fit = col[:array.shape[1]]

    return pd.DataFrame(array, columns=col_fit)
