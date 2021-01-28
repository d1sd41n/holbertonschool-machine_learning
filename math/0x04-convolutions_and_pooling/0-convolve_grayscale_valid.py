#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """[summary]

    Args:
        images ([type]): [description]
        kernel ([type]): [description]

    Returns:
        [type]: [description]
    """
    m, h, w = images.shape

    z_h = h - kernel.shape[0] + 1
    z_w = w - kernel.shape[1] + 1

    z = np.zeros((m, z_h, z_w))

    for i in range(z_h):
        for j in range(z_w):
            z[:, i, j] = (kernel * images[:,
                                          i: i + kernel.shape[0],
                                          j: j + kernel.shape[1]]
                          ).sum(axis=(1, 2))
    return z
