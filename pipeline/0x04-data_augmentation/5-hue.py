#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def change_hue(image, delta):
    """[summary]

    Args:
        image ([type]): [description]
        delta ([type]): [description]

    Returns:
        [type]: [description]
    """
    adjusted = tf.image.adjust_hue(image, delta)
    return adjusted
