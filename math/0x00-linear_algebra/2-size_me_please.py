#!/usr/bin/env python3
"""
here some unnecessary documentation :)
"""


def matrix_shape(matrix):
    """returns sape of matrix"""
    m = []
    while type(matrix) == list:
        m.append(len(matrix))
        matrix = matrix[0]
    return m
