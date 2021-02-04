#!/usr/bin/env python3
"""[summary]

Returns:
    [type]: [description]
"""
import tensorflow.keras as K


def lenet5(X):
    """[summary]

    Args:
        X ([type]): [description]

    Returns:
        [type]: [description]
    """
    he_ini = K.initializers.he_normal()
    l1_conv = K.layers.Conv2D(filters=6,
                              kernel_size=(5, 5),
                              padding='same',
                              activation='relu',
                              kernel_initializer=he_ini)(X)
    l1_pool = K.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2))(l1_conv)

    l2_conv = K.layers.Conv2D(filters=16,
                              kernel_size=(5, 5), padding='valid',
                              activation='relu',
                              kernel_initializer=he_ini)(l1_pool)
    l2_pool = K.layers.MaxPool2D(pool_size=(2, 2),
                                 strides=(2, 2))(l2_conv)

    flat = K.layers.Flatten()(l2_pool)

    l3_fully = K.layers.Dense(units=120,
                              activation='relu',
                              kernel_initializer=he_ini)(flat)

    l4_fully = K.layers.Dense(units=84,
                              activation='relu',
                              kernel_initializer=he_ini)(l3_fully)

    l5_fully = K.layers.Dense(units=10,
                              activation='softmax',
                              kernel_initializer=he_ini)(l4_fully)

    nn = K.Model(inputs=X, outputs=l5_fully)
    nn.compile(optimizer=K.optimizers.Adam(),
               loss='categorical_crossentropy',
               metrics=['accuracy'])
    return nn
