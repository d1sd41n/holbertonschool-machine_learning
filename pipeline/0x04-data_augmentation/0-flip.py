#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def flip_image(image):
    """[summary]

    Args:
        image ([type]): [description]

    Returns:
        [type]: [description]
    """
    fliped_img = tf.image.flip_left_right(image)
    return fliped_img
