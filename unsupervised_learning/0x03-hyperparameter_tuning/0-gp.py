#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


class GaussianProcess:
    """[summary]
    """

    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """[summary]

        Args:
            X_init ([type]): [description]
            Y_init ([type]): [description]
            l (int, optional): [description]. Defaults to 1.
            sigma_f (int, optional): [description]. Defaults to 1.
        """
        self.X = X_init

        self.Y = Y_init
        self.l = l

        self.sigma_f = sigma_f

        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """[summary]

        Args:
            X1 ([type]): [description]
            X2 ([type]): [description]

        Returns:
            [type]: [description]
        """

        sqdist = np.sum(X1 ** 2, 1).reshape(
            -1, 1) + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

        return self.sigma_f ** 2 * np.exp(
            -0.5 / self.l ** 2 * sqdist)
