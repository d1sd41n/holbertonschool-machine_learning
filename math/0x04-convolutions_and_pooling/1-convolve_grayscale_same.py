#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """[summary]

    Args:
        images ([type]): [description]
        kernel ([type]): [description]

    Returns:
        [type]: [description]
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = max((kh - 1) // 2,
             kh // 2)
    pw = max((kw - 1) // 2,
             kw // 2)

    padding = np.pad(images, ((0, 0),
                              (ph, ph),
                              (pw, pw)))
    z = np.zeros((m, h, w))
    for i in range(h):
        for j in range(w):
            z[:, i, j] = (kernel * padding[:,
                                           i: i + kh,
                                           j: j + kw]
                          ).sum(axis=(1, 2))
    return z
