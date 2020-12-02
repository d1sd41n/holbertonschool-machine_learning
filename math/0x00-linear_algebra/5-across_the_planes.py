#!/usr/bin/env python3
"""
here some unnecessary documentation :)
"""


def add_matrices2D(mat1, mat2):
    """
    Adds two matrices
    """
    summ = []
    if len(mat1) != len(mat2):
        return None
    for i in range(0, len(mat1)):
        if len(mat1[i]) == len(mat2[i]):
            r = []
            for j in range(0, len(mat1[i])):
                r.append(mat1[i][j] + mat2[i][j])
            summ.append(r)
        else:
            return None
    return summ
