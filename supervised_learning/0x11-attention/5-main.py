#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""

import numpy as np


def get_angles(pos, i, d_model):
    """[summary]

    Args:
        pos ([type]): [description]
        i ([type]): [description]
        d_model ([type]): [description]

    Returns:
        [type]: [description]
    """
    angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(max_seq_len, dm):
    """[summary]

    Args:
        max_seq_len ([type]): [description]
        dm ([type]): [description]

    Returns:
        [type]: [description]
    """
    pos_encoding = get_angles(np.arange(max_seq_len)[:, np.newaxis],
                              np.arange(dm)[np.newaxis, :],
                              dm)
    pos_encoding[:, 0::2] = np.sin(pos_encoding[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(pos_encoding[:, 1::2])
    return pos_encoding
