#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """[summary]

    Args:
        images ([type]): [description]
        kernel ([type]): [description]
        padding (str, optional): [description]. Defaults to 'same'.
        stride (tuple, optional): [description]. Defaults to (1, 1).

    Returns:
        [type]: [description]
    """
    m, h, w, c = images.shape
    kh, kw, _, nc = kernels.shape
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

    z = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)))
    z_h = int((h + 2 * ph - kh) / sh + 1)
    z_w = int((w + 2 * pw - kw) / sw + 1)

    z_ = np.zeros((m, z_h, z_w, nc))

    for i in range(z_h):
        for j in range(z_w):
            for k in range(nc):
                z_[:, i, j, k] = (kernels[:, :, :, k] * z[:,
                                                          i * sh: i * sh + kh,
                                                          j * sw: j * sw + kw,
                                                          :]
                                  ).sum(axis=(1, 2, 3))
    return z_
