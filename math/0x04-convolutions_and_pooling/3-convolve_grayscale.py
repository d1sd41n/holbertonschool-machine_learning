#!/usr/bin/env python3
""" This module contains the function convolve_grayscale. """
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
    Performs a convolution on grayscale images.
    images is a numpy.ndarray with shape (m, h, w) containing multiple
        grayscale images.
        m is the number of images.
        h is the height in pixels of the images.
        w is the width in pixels of the images.
    kernel is a numpy.ndarray with shape (kh, kw) containing the kernel for the
     convolution.
        kh is the height of the kernel.
        kw is the width of the kernel.
    padding is either a tuple of (ph, pw), same, or valid.
        if same, performs a same convolution.
        if valid, performs a valid convolution.
        if a tuple:
            ph is the padding for the height of the image.
            pw is the padding for the width of the image.
        the image is padded with 0s.
    stride is a tuple of (sh, sw).
        sh is the stride for the height of the image.
        sw is the stride for the width of the image.
    Returns: a numpy.ndarray containing the convolved images.
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

    padded = np.pad(images, ((0,), (ph,), (pw,)))
    ansh = int((h + 2 * ph - kh) / sh + 1)
    answ = int((w + 2 * pw - kw) / sw + 1)
    ans = np.zeros((m, ansh, answ))
    for i in range(ansh):
        for j in range(answ):
            x = i * sh
            y = j * sw
            ans[:, i, j] = (padded[:, x: x + kh, y: y + kw] *
                            kernel).sum(axis=(1, 2))
    return ans
