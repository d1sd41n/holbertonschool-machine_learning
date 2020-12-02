#!/usr/bin/env python3
"""
here some unnecessary documentation :)
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two matrices"""
    if len(mat1) != len(mat2) and axis == 1:
        return None
    if len(mat1[0]) != len(mat2[0]) and axis == 0:
        return None
    m = []
    for row in range(0, len(mat1)):
        r = []
        for cl in range(0, len(mat1[0])):
            r.append(mat1[row][cl])
        if axis == 1:
            for cl in range(0, len(mat2[0])):
                r.append(mat2[row][cl])
        m.append(r)
    if axis == 0:
        for row in range(0, len(mat2)):
            r = []
            for cl in range(0, len(mat2[0])):
                r.append(mat2[row][cl])
            m.append(r)
    return m
