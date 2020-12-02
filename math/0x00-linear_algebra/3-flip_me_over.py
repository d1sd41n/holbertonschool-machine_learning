#!/usr/bin/env python3
"""
here some unnecessary documentation :)
"""


def matrix_transpose(matrix):
    """transposes a matrix"""

    c = len(matrix)
    r = len(matrix[0])
    t_m = []
    for i in range(0, r):
        aux_r = []
        for j in range(0, c):
            aux_r.append(matrix[j][i])
        t_m.append(aux_r)
    return t_m
