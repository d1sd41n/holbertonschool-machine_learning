#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """[summary]

    Args:
        images ([type]): [description]
        kernel_shape ([type]): [description]
        stride ([type]): [description]
        mode (str, optional): [description]. Defaults to 'max'.

    Returns:
        [type]: [description]
    """
    m, h, w, c = images.shape

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    z_h = int(((h - kh) / sh) + 1)
    z_w = int(((w - kw) / sw) + 1)

    z = np.zeros((m, z_h, z_w, c))
    z_m = np.arange(0, m)

    for i in range(z_h):
        for j in range(z_w):
            if mode == 'max':
                data = np.max(images[z_m,
                                     i*sh:kh+(i*sh),
                                     j*sw:kw+(j*sw)],
                              axis=(1, 2))
            if mode == 'avg':
                data = np.mean(images[z_m,
                                      i*sh:kh+(i*sh),
                                      j*sw:kw+(j*sw)],
                               axis=(1, 2))
            z[z_m, i, j] = data
    return z
