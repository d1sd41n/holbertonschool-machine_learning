#!/usr/bin/env python3
"""
docs
"""


class Poisson:
    """docs"""

    def __init__(self, data=None, lambtha=1.):
        """docs"""
        if data is not None:
            if len(data) <= 2:
                raise ValueError("data must contain multiple values")
            elif type(data) != list:
                raise TypeError("data must be a list")
            else:
                self.lambtha = float(sum(data) / len(data))
        else:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
