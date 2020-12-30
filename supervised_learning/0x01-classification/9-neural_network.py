#!/usr/bin/env python3
"""[summary]

Raises:
    TypeError: [description]
    ValueError: [description]
    TypeError: [description]
    ValueError: [description]

Returns:
    [type]: [description]
"""
import numpy as np


class NeuralNetwork:
    """[summary]
    """

    def __init__(self, nx, nodes):
        """[summary]

        Args:
            nx ([type]): [description]
            nodes ([type]): [description]

        Raises:
            TypeError: [description]
            ValueError: [description]
            TypeError: [description]
            ValueError: [description]
        """

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        # layer 1
        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0

        # layer 2
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__W1

    @property
    def b1(self):
        """[summary]

        Returns:
            [type]: [description]
        """""
        return self.__b1

    @property
    def A1(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__A1

    @property
    def W2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__W2

    @property
    def b2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__b2

    @property
    def A2(self):
        """[summary]

        Returns:
            [type]: [description]
        """
        return self.__A2
