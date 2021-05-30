import numpy as np
import math
import tensorflow as tf


# Reshape data from (,784) to (,28,28)
def reshape_data(x_train, x_test):
    img_dim = int(math.sqrt(x_train.shape[1]))

    x_train = np.reshape(x_train, newshape=(x_train.shape[0], img_dim, img_dim))
    x_test = np.reshape(x_test, newshape=(x_test.shape[0], img_dim, img_dim))

    return x_train, x_test


# Cast data from 'uint8' to 'tf.float32'/'tf.int32
def cast_datatype(x_train, y_train, x_test, y_test):
    # Dimension of data
    print('X_train shape: {} \t X_train type: {}'.format(x_train.shape, x_train.dtype))
    print('y_train shape: {} \t y_train type: {}'.format(y_train.shape, y_train.dtype))
    print('X_test shape: {} \t X_test type: {}'.format(x_test.shape, x_test.dtype))
    print('y_test shape: {} \t y_test type: {}'.format(y_test.shape, y_test.dtype))

    # Cast data
    x_train = tf.cast(x_train, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.int32)
    x_test = tf.cast(x_test, dtype=tf.float32)
    y_test = tf.cast(y_test, dtype=tf.int32)

    return x_train, y_train, x_test, y_test


# 1.3. Add channel dimension (NHWC). Used in CNN
def expand_dims(x_train, x_test):
    x_train = tf.expand_dims(x_train, axis=3)
    x_test = tf.expand_dims(x_test, axis=3)
    print('X_train shape: {} \t X_train type: {}'.format(x_train.shape, x_train.dtype))
    print('X_test shape: {} \t X_test type: {}'.format(x_test.shape, x_test.dtype))

    return x_train, x_test
