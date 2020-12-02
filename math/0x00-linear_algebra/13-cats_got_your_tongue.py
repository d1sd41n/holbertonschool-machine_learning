#!/usr/bin/env python3
"""
here some unnecessary documentation :)
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """concatenates two matrices"""
    return np.concatenate((mat1, mat2), axis)
