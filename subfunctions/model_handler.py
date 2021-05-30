import os
import tensorflow.keras.optimizers as opt
import tensorflow as tf
from tensorflow.keras import layers


def generate_cnn_model():
    # Index       Layer           Channels    H*W
    # Layer 0     Input           1           28*28
    # Layer 1     Conv (ReLU)     16          14*14
    # Layer 2     Conv (ReLU)     32          7*7
    # Layer 3     Conv (ReLU)     64          4*4
    # Layer 4     Max-Pool        64          1*1
    # Layer 5     Output          10          1*1

    model = tf.keras.Sequential(name='Model')
    model.add(
        layers.Conv2D(16, kernel_size=3, strides=2, padding='same', data_format='channels_last', activation='relu',
                      input_shape=(28, 28, 1), name='Conv01', ))
    model.add(
        layers.Conv2D(32, kernel_size=3, strides=2, padding='same', data_format='channels_last', activation='relu',
                      name='Conv02'))
    model.add(
        layers.Conv2D(64, kernel_size=3, strides=2, padding='same', data_format='channels_last', activation='relu',
                      name='Conv03'))
    model.add(layers.GlobalMaxPool2D(data_format='channels_last', name='max_pool'))
    model.add(layers.Dense(3, activation='softmax', name='Output'))

    model.summary()
    return model


def generate_model_mlp_w_dropout(input_shape):
    model = tf.keras.Sequential(name='Model')
    model.add(layers.Dropout(0.3, input_shape=(input_shape,), name='Dropout_Input'))
    model.add(layers.Dense(128, activation='relu', name='Dense_1'))
    # model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,), name='Dense_1'))
    model.add(layers.Dropout(0.3, name='Dropout_1'))
    model.add(layers.Dense(64, activation='relu', name='Dense_2'))
    model.add(layers.Dropout(0.2, name='Dropout_2'))
    model.add(layers.Dense(32, activation='relu', name='Dense_3'))
    model.add(layers.Dropout(0.2, name='Dropout_3'))
    model.add(layers.Dense(3, activation='softmax', name='Output'))

    model.summary()
    return model


def generate_model_mlp(input_shape):
    model = tf.keras.Sequential(name='Model')
    model.add(layers.Dense(128, activation='relu', input_shape=(input_shape,), name='Dense_1'))
    model.add(layers.Dense(64, activation='relu', name='Dense_2'))
    model.add(layers.Dense(3, activation='softmax', name='Output'))

    model.summary()
    return model


def get_model_optimizer(optimizer_str, learning_rate):
    if optimizer_str == "SGD":
        optimizer = opt.SGD(lr=learning_rate, momentum=0.0, nesterov=False, name=optimizer_str)
    elif optimizer_str == "SGD_with_momentum":
        optimizer = opt.SGD(lr=learning_rate * 5, momentum=0.8, decay=learning_rate / 10, nesterov=False, name=optimizer_str)
    elif optimizer_str == "RMSprop":
        optimizer = opt.RMSprop(
            lr=learning_rate,
            rho=0.9,
            momentum=0.0,
            epsilon=1e-07,
            centered=False,
            name=optimizer_str
        )
    elif optimizer_str == "Adam":
        optimizer = opt.Adam(lr=learning_rate, name=optimizer_str)

    return optimizer


def save_model(model, model_name):
    model_folder = 'content/savedModels/models/'
    # model_name = 'model00'

    if not os.path.exists(model_folder):
        os.mkdir(model_folder)

    # model.save_weights(model_folder + model_name)
    model.save_weights(model_folder + model_name + '.h5', save_format='h5')
