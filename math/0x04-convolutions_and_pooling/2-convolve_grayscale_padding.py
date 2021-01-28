#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """[summary]

    Args:
        images ([type]): [description]
        kernel ([type]): [description]
        padding ([type]): [description]

    Returns:
        [type]: [description]
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph, pw = padding
    padded = np.pad(images, ((0, 0),
                             (ph, ph),
                             (pw, pw)))

    z_h = h + (2 * ph) - kh + 1
    z_w = w + (2 * pw) - kw + 1

    z = np.zeros((m, z_h, z_w))
    for i in range(z_h):
        for j in range(z_w):
            z[:, i, j] = (kernel * padded[:,
                                          i: i + kh,
                                          j: j + kw]
                          ).sum(axis=(1, 2))
    return z
