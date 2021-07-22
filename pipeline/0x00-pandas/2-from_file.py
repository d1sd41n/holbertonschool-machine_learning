#!/usr/bin/env python3
"""[summary]
"""
import pandas as pd


def from_file(filename, delimiter):
    """[summary]

    Args:
        filename ([type]): [description]
        delimiter ([type]): [description]

    Returns:
        [type]: [description]
    """
    return pd.read_csv(filename, sep=delimiter)
