#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """[summary]

    Args:
        max_seq_len ([type]): [description]
        dm ([type]): [description]

    Returns:
        [type]: [description]
    """
    f = 1 / (np.power(10000,
                      (2*(np.arange(dm
                                    )[np.newaxis, :
                                        ]//2)/np.float32(
                                            dm))))
    Wt = (f * np.arange(max_seq_len)[:, np.newaxis])
    tp = np.zeros(
        (max_seq_len, dm)
        )
    tp[:, 0::2] = np.sin(Wt[:, 0::2])
    tp[:, 1::2] = np.cos(Wt[:, 1::2])
    return tp
