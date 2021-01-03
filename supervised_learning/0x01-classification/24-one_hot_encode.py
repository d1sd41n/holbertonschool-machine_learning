#!/usr/bin/env python3
"""Contains the one_hot_encode method"""

import numpy as np


def one_hot_encode(Y, classes):
    """[summary]

    Returns:
        [type]: [description]
    """
    if (not isinstance(Y, np.ndarray) or len(Y) == 0 or
            not isinstance(classes, int) or classes <= np.amax(Y)):
        return None
    return np.eye(classes)[Y].T
