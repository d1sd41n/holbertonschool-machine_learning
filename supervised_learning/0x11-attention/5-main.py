#!/usr/bin/env python3
"""Function sdp_attention."""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """[summary]

    Args:
        Q ([type]): [description]
        K ([type]): [description]
        V ([type]): [description]
        mask ([type], optional): [description]. Defaults to None.

    Returns:
        [type]: [description]
    """
    qpk = tf.matmul(Q, K,
                    transpose_b=True) / tf.sqrt(
                        tf.cast(tf.shape(K)[-1], tf.float32))
    if mask is not None:
        qpk += mask * -1e9
    weights = tf.nn.softmax(
        qpk,
        axis=-1
        )
    output = tf.matmul(weights, V)
    return output, weights
