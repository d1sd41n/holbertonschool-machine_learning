#!/usr/bin/env python3
"""
here some unnecessary documentation :)
"""


def np_slice(matrix, axes={}):
    """
    slices the matrix
    """
    aux = []
    for row in range(0, len(matrix.shape)):
        if axes.get(row, None):
            aux.append(slice(*axes.get(row, None)))
            continue
        aux.append(slice(None))
    return matrix[tuple(aux)]
