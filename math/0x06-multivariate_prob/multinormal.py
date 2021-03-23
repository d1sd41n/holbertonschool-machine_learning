#!/usr/bin/env python3
"""[summary]

Raises:
    TypeError: [description]
    ValueError: [description]
"""
import numpy as np


class MultiNormal:
    """[summary]
    """

    def __init__(self, data):
        """[summary]

        Args:
            data ([type]): [description]

        Raises:
            TypeError: [description]
            ValueError: [description]

        """
        if type(data) != np.ndarray or len(data.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')
        if data.shape[1] < 2:
            raise ValueError('data must contain multiple data points')
        self.mean = np.mean(data, axis=1).reshape(data.shape[0], 1)
        self.cov = np.matmul((data - self.mean),
                             (data - self.mean).T) / (data.shape[1] - 1)
