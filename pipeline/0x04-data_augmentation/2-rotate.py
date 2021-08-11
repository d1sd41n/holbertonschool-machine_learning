#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def rotate_image(image):
    """[summary]

    Args:
        image ([type]): [description]

    Returns:
        [type]: [description]
    """
    rotated = tf.image.rot90(image, k=1)

    return rotated
