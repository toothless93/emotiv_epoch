import datetime
import tensorflow as tf


def enable_tensorboard_callback(gradient_method, patience):
    log_dir = "content/savedModels/fit/"
    callback_log_dir = log_dir + gradient_method + "-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=callback_log_dir, histogram_freq=1)

    tensorboard_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True
    )
    return tensorboard_callback
