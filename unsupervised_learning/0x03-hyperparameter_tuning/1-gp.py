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
        """
        Isotropic squared exponential kernel.
        - X1: Array of m points (m x d).
        - X2: Array of n points (n x d).
        Returns:
            (m x n) kernel matrix.
        """

        sqdist = np.sum(X1 ** 2,
                        1).reshape(-1, 1) \
            + np.sum(X2 ** 2, 1) - 2 * np.dot(X1, X2.T)

        return self.sigma_f ** 2 * np.exp(
            -0.5 / self.l ** 2 * sqdist)

    def predict(self, X_s):
        """[summary]

        Args:
            X_s ([type]): [description]

        Returns:
            [type]: [description]
        """
        
        Y_train = self.Y
        X_train = self.X
        
        sigma_y = 0
        K = self.K
        K_s = self.kernel(X_train, X_s)
        
        K_ss = self.kernel(X_s, X_s)
        K_inv = np.linalg.inv(K)
        
        m_se = K_s.T.dot(K_inv).dot(Y_train)
        
        m_se = m_se.reshape(-1)
        cov_s = np.diag(K_ss - K_s.T.dot(
            K_inv).dot(K_s))
        
        return m_se, cov_s
