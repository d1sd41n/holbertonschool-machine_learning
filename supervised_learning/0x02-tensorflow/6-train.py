#!/usr/bin/env python3

import tensorflow as tf
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_placeholders = __import__('0-create_placeholders').create_placeholders
create_train_op = __import__('5-create_train_op').create_train_op
forward_prop = __import__('2-forward_prop').forward_prop


def train(X_train, Y_train, X_valid, Y_valid,
          layer_sizes, activations, alpha, iterations,
          save_path="/tmp/model.ckpt"):
    """[summary]

    Args:
        X_train ([type]): [description]
        Y_train ([type]): [description]
        X_valid ([type]): [description]
        Y_valid ([type]): [description]
        layer_sizes ([type]): [description]
        activations ([type]): [description]
        alpha ([type]): [description]
        iterations ([type]): [description]
        save_path (str, optional): [description]. Defaults to "/tmp/model.ckpt".

    Returns:
        [type]: [description]
    """

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    accuracy = calculate_accuracy(y, y_pred)
    loss = calculate_loss(y, y_pred)
    train_op = create_train_op(loss, alpha)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.Saver()
        for i in range(iterations):
            _, accur_train, loss_value_train = sess.run(
                (train_op, accuracy, loss), feed_dict={x: X_train, y: Y_train})
            _, accur_val, loss_value_val = sess.run(
                (train_op, accuracy, loss), feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(loss_value_train))
                print("\tTraining Accuracy: {}".format(accur_train))
                print("\tValidation Cost: {}".format(loss_value_val))
                print("\tValidation Accuracy: {}".format(accur_val))
        saver.save(sess, save_path)
    return save_path
