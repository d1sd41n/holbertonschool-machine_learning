#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """[summary]

    Args:
        tf ([type]): [description]
    """

    def __init__(self, vocab, embedding, units, batch):
        """[summary]

        Args:
            vocab ([type]): [description]
            embedding ([type]): [description]
            units ([type]): [description]
            batch ([type]): [description]
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab,
            embedding
        )
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer='glorot_uniform',
            return_sequences=True,
            return_state=True
        )
        self.F = tf.keras.layers.Dense(
            vocab
        )

    def call(self, x, s_prev, hidden_states):
        """[summary]

        Args:
            x ([type]): [description]
            s_prev ([type]): [description]
            hidden_states ([type]): [description]

        Returns:
            [type]: [description]
        """
        _, u = s_prev.shape
        t, _ = SelfAttention(
            u
        )(
            s_prev,
            hidden_states
        )
        out_1, h = self. \
            gru(
                tf.concat(
                    [tf.expand_dims(t,
                                    1
                                    ),
                     self.embedding(
                         x
                    )
                    ],
                    axis=-1
                )
            )
        out_1 = tf.reshape(
            out_1,
            (out_1.shape[0],
             out_1.shape[2]))
        return self.F(out_1), h
