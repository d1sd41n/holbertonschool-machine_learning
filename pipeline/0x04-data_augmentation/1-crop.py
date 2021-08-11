#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def crop_image(image, size):
    """[summary]

    Args:
        image ([type]): [description]
        size ([type]): [description]

    Returns:
        [type]: [description]
    """
    croped = tf.random_crop(image, size)

    return croped
