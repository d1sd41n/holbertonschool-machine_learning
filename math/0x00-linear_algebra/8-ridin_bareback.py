#!/usr/bin/env python3
"""
here some unnecessary documentation :)
"""


def mat_mul(mat1, mat2):
    """
    matrix mul
    """
    m = []
    if len(mat2) != len(mat1[0]):
        return None
    for i in range(0, len(mat1)):
        r = []

        for c in range(0, len(mat2[0])):
            aux = 0

            for j in range(0, len(mat1[0])):

                n = mat1[i][j] * mat2[j][c]
                aux = aux + n
            r.append(aux)

        m.append(r)
    return m
