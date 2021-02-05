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
    # he_ini = K.initializers.he_normal()
    # con_l1 = K.layers.Conv2D(filters=6,
    #                          kernel_size=(5, 5),
    #                          padding='same',
    #                          activation='relu',
    #                          kernel_initializer=he_ini)(X)
    # l1_pool = K.layers.MaxPool2D(pool_size=(2, 2),
    #                              strides=(2, 2))(con_l1)
    # l2_con = K.layers.Conv2D(filters=16,
    #                          kernel_size=(5, 5), padding='valid',
    #                          activation='relu',
    #                          kernel_initializer=he_ini)(l1_pool)
    # l2_pool = K.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(l2_con)
    # flat = K.layers.Flatten()(l2_pool)
    # l3_fully = K.layers.Dense(units=120,
    #                           activation='relu',
    #                           kernel_initializer=he_ini)(flat)
    # l4_fully = K.layers.Dense(units=84,
    #                           activation='relu',
    #                           kernel_initializer=he_ini)(l3_fully)
    # l5_fully = K.layers.Dense(units=10,
    #                           activation='softmax',
    #                           kernel_initializer=he_ini)(l4_fully)
    # nn = K.Model(inputs=X, outputs=l5_fully)
    # nn.compile(optimizer=K.optimizers.Adam(),
    #            loss='categorical_crossentropy',
    #            metrics=['accuracy'])
    # return nn
    activation = 'relu'
    kInit = K.initializers.he_normal(seed=None)

    layer_1 = K.layers.Conv2D(filters=6, kernel_size=5,
                              padding='same',
                              activation=activation,
                              kernel_initializer=kInit)(X)

    pool_1 = K.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(layer_1)

    layer_2 = K.layers.Conv2D(filters=16, kernel_size=5,
                              padding='valid',
                              activation=activation,
                              kernel_initializer=kInit)(pool_1)

    pool_2 = K.layers.MaxPooling2D(pool_size=[2, 2],
                                   strides=2)(layer_2)

    flatten = K.layers.Flatten()(pool_2)

    layer_3 = K.layers.Dense(120, activation=activation,
                             kernel_initializer=kInit)(flatten)

    layer_4 = K.layers.Dense(84, activation=activation,
                             kernel_initializer=kInit)(layer_3)

    output_layer = K.layers.Dense(10, activation='softmax',
                                  kernel_initializer=kInit)(layer_4)

    model = K.models.Model(X, output_layer)

    model.compile(optimizer=K.optimizers.Adam(),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model
