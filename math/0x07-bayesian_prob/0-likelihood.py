#!/usr/bin/env python3
"""[summary]

Raises:
    ValueError: [description]
    ValueError: [description]
    ValueError: [description]
    TypeError: [description]
    ValueError: [description]

Returns:
    [type]: [description]
"""
import numpy as np


def likelihood(x, n, P):
    """[summary]

    Args:
        x ([type]): [description]
        n ([type]): [description]
        P ([type]): [description]

    Raises:
        ValueError: [description]
        ValueError: [description]
        ValueError: [description]
        TypeError: [description]
        ValueError: [description]

    Returns:
        [type]: [description]
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError('n must be a positive integer')
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            'x must be an integer that is greater than or equal to 0')
    if x > n:
        raise ValueError('x cannot be greater than n')
    if not isinstance(P, np.ndarray) or len(P.shape) != 1:
        raise TypeError('P must be a 1D numpy.ndarray')
    if np.any(P < 0) or np.any(P > 1):
        raise ValueError('All values in P must be in the range [0, 1]')
    return (np.math.factorial(n) /
            (np.math.factorial(x) * np.math.factorial(n - x)
             )) * (P ** x) * (1 - P) ** (n - x)
