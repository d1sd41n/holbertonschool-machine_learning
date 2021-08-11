#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def change_brightness(image, max_delta):
    """[summary]

    Args:
        image ([type]): [description]
        max_delta ([type]): [description]

    Returns:
        [type]: [description]
    """
    adjusted = tf.image.adjust_brightness(image, max_delta)

    return adjusted
