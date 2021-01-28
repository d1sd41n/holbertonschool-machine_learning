#!/usr/bin/env python3
"""[summary]

    Returns:
        [type]: [description]
    """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """[summary]

    Args:
        images ([type]): [description]
        kernel ([type]): [description]
        padding (str, optional): [description]. Defaults to 'same'.
        stride (tuple, optional): [description]. Defaults to (1, 1).

    Returns:
        [type]: [description]
    """
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if padding == 'same':
        ph = int(((h - 1) * sh - h + kh) / 2) + 1
        pw = int(((w - 1) * sw - w + kw) / 2) + 1
    if padding == 'valid':
        ph = 0
        pw = 0
    if type(padding) is tuple:
        ph = padding[0]
        pw = padding[1]

    p_m = np.pad(images, ((0, 0),
                          (ph, ph),
                          (pw, pw)))

    a_h = int((h + 2 * ph - kh) / sh + 1)
    a_w = int((w + 2 * pw - kw) / sw + 1)

    z = np.zeros((m, a_h, a_w))
    for i in range(a_h):
        for j in range(a_w):
            z[:, i, j] = (kernel * p_m[:,
                                       i * sh: i * sh + kh,
                                       j * sw: j * sw + kw]
                          ).sum(axis=(1, 2))
    return z
