#!/usr/bin/env python3
"""[summary]

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """
import numpy as np


def definiteness(matrix):
    """[summary]

    Args:    ss
        matrix ([type]): [description]

    Raises:
        TypeError: [description]

    Returns:
        [type]: [description]
    """
    if type(matrix) is not np.ndarray:
        raise TypeError("matrix must be a numpy.ndarray")
    if (matrix.ndim != 2 or
        matrix.shape[0] != matrix.shape[1] or
        matrix.shape[1] == 0 or
            (np.matrix.conjugate(matrix.T) != matrix).any()):
        return None
    wight, _ = np.linalg.eig(matrix)

    if np.all(wight > 0):
        return 'Positive definite'
    if np.all(wight >= 0):
        return 'Positive semi-definite'
    if np.all(wight < 0):
        return 'Negative definite'
    if np.all(wight <= 0):
        return 'Negative semi-definite'
    else:
        return 'Indefinite'
