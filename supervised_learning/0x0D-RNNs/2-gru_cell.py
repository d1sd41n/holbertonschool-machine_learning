#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import numpy as np


def softmax(x):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return np.exp(x) / np.sum(
        np.exp(x), axis=1, keepdims=True)


def sigmoid(x):
    """[summary]

    Args:
        x ([type]): [description]

    Returns:
        [type]: [description]
    """
    return 1 / (1 + np.exp(-x))


class GRUCell:
    """[summary]
    """

    def __init__(self, i, h, o):
        """[summary]

        Args:
            i ([type]): [description]
            h ([type]): [description]
            o ([type]): [description]
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """[summary]

        Args:
            h_prev ([type]): [description]
            x_t ([type]): [description]

        Returns:
            [type]: [description]
        """
        matrix = np.concatenate((h_prev, x_t), axis=1)
        z_t = sigmoid(np.matmul(matrix, self.Wz) + self.bz)
        r_t = sigmoid(np.matmul(matrix, self.Wr) + self.br)
        matrix2 = np.concatenate((r_t * h_prev, x_t), axis=1)
        prime_h = np.tanh(np.matmul(matrix2, self.Wh) + self.bh)
        h_next = (1 - z_t) * h_prev + z_t * prime_h
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)
        return h_next, y
