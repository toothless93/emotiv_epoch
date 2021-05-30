import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.utils.np_utils import to_categorical
from sklearn.preprocessing import StandardScaler


def preprocess_inputs(df, target, train_size, multi_users):
    df = df.copy()

    if multi_users:
        # One-hot encode whichever target column is not being used
        targets = ['Class', 'User']
        targets.remove(target)
        df = onehot_encode(df, column=targets[0])

    # Split df into x and y
    y = df[target].copy()
    x = df.drop(target, axis=1)
    if not multi_users:
        x = x.drop('User', axis=1)

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_size, random_state=42)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    # Scale X with a standard scaler
    scaler = StandardScaler()
    scaler.fit(x_train)

    x_train = pd.DataFrame(scaler.transform(x_train), columns=x.columns)
    x_test = pd.DataFrame(scaler.transform(x_test), columns=x.columns)

    return x_train, x_test, y_train, y_test


def onehot_encode(dataframe, column):
    df = dataframe.copy()
    lb_onehot = pd.get_dummies(df[column], prefix=column)
    df = pd.concat([df, lb_onehot], axis=1)
    df = df.drop(column, axis=1)

    return df
