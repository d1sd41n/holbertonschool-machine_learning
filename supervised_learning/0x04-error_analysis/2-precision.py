#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def precision(confusion):
    """[summary]

    Args:
        confusion ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.diagonal(confusion) / np.sum(confusion, axis=0)
