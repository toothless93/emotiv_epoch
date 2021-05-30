import tensorflow as tf
import pandas as pd
from subfunctions import visualization, model_handler, callbacks, data_handler, file_handler, preprocess
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
import gc
gc.enable()


if __name__ == '__main__':
    # These folders contain the experimental results:
    # Folder                        Functionality
    # "content"                     Contain all the project's results
    # "content/dataVisual"          Contain Sprite image, which visualize the dataset
    # "content/savedModels"         Contain the all the info of the training models & processes
    # "content/savedModels/models"  Contain the trained models
    # "content/savedModels/fit"     Contain the history of training processes
    file_handler.create_folders(["content", "content/dataVisual",
                                 "content/savedModels", "content/savedModels/models", "content/savedModels/fit"])

    # STEP 1: READ DATASET
    dataset_users = [pd.read_csv('dataset/user_' + user + '.csv', delimiter=',') for user in ['a', 'b', 'c', 'd']]
    [dataset_a, dataset_b, dataset_c, dataset_d] = dataset_users

    # print('Single user dataset shape: ', dataset_a.shape)
    # print(dataset_a.head())
    # print(dataset_b.head())
    # print(dataset_c.head())
    # print(dataset_d.head())

    for i in range(len(dataset_users)):
        dataset_users[i]['User'] = pd.Series(i, index=dataset_users[i].index)
    dataset = pd.concat(dataset_users, axis=0).sample(frac=1.0, random_state=123).reset_index(drop=True)
    print('All users dataset shape: ', dataset.shape)

    dataset = dataset_a
    dataset.dataframeName = 'dataset.csv'

    # STEP 2: DATA EXPLORATION
    # target = 'Class'
    # col = dataset.columns
    # features = col[1:]
    # print(features)
    # print(dataset[target].value_counts())

    # visualization.plotScatterMatrix(dataset, 20, 10)
    # visualization.plotCorrelationMatrix(dataset, 25)
    # visualization.plotPerColumnDistribution(dataset, 10, 5)
    # visualization.plot_sensors_correlation(threshold_value=.97, dataset=dataset)

    # STEP 3: DATA ANALYSIS
    x_train, x_test, y_train, y_test = preprocess.preprocess_inputs(df=dataset, target='Class', train_size=0.8, multi_users=False)
    # print(x_train.head())
    # print(y_train.value_counts())

    # MLP MODEL WITH DROPOUT
    # STEP : BUILD MODEL
    model = model_handler.generate_model_mlp_w_dropout(x_train.shape[1])

    # STEP : BUILD & TRAIN MODEL
    # 3.1. Hyper-parameter selection
    EPOCHS, BATCH_SIZE = 600, 32
    LEARNING_RATE = 1e-3

    # step = tf.Variable(0, trainable=False)
    # num_instance = 1856
    # bounds = [10*num_instance//BATCH_SIZE, 50*num_instance//BATCH_SIZE, 100*num_instance//BATCH_SIZE]
    # values = [4e-4, 4e-5, 4e-5, 4e-5]
    # learning_rate_fn = PiecewiseConstantDecay(bounds, values)
    # LEARNING_RATE = learning_rate_fn(step)

    GRADIENT_METHOD = "Adam" # 4 optimizers: "SGD", "SGD_with_momentum", "RMSprop", "Adam".
    PATIENCE = 40
    optimizer = model_handler.get_model_optimizer(GRADIENT_METHOD, LEARNING_RATE)
    tensorboard_callback = callbacks.enable_tensorboard_callback(GRADIENT_METHOD, PATIENCE)
    LOSS = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # 3.2. Compile and train model
    model.compile(optimizer=optimizer, loss=LOSS, metrics='accuracy')
    history = model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_split=0.2,
                        callbacks=[tensorboard_callback])

    # # 3.3. Save NN model
    # model_name = "MLP_" + gradient_method
    # model_handler.save_model(model, model_name)

    # 3.4. Plot the result
    visualization.plot_accuracy_loss(training_history=history)

    # STEP 4: EVALUATE
    model.evaluate(x_test, y_test, batch_size=64)

