#!/usr/bin/env python3
"""
docs
"""


def poly_derivative(poly):
    """
    some docs
    """
    if type(poly) != list:
        return None
    elif len(poly) == 1:
        return [0]
    elif type(poly[0]) not in [int, float]:
        return None
    else:
        x = 0
        d_arr = []

        while x < len(poly):

            if type(poly[x]) not in (int, float):
                return None

            elif len(poly) == 1:
                d_arr.append(0)

            else:
                if x == 0:
                    x += 1
                    continue

                d_arr.append(x*poly[x])
            x += 1

        return d_arr
