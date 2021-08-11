#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def shear_image(image, intensity):
    """[summary]

    Args:
        image ([type]): [description]
        intensity ([type]): [description]

    Returns:
        [type]: [description]
    """
    img = tf.keras.preprocessing.image.img_to_array(image)
    sheared = tf.keras.preprocessing.image.random_shear(img, intensity,
                                                        row_axis=0,
                                                        col_axis=1,
                                                        channel_axis=2)
    sheared_img = tf.keras.preprocessing.image.array_to_img(sheared)
    return sheared_img