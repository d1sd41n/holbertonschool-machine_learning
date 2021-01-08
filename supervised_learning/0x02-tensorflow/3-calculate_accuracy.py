#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """[summary]

    Args:
        y ([type]): [description]
        y_pred ([type]): [description]

    Returns:
        [type]: [description]
    """
    y_hot = tf.argmax(y, axis=1)
    y_p_hot = tf.argmax(y_pred, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(y_hot, y_p_hot), dtype=tf.float32))
