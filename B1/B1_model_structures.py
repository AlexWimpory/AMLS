from tensorflow.python.keras.layers import *


def model_1(num_labels):
    return [
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='valid', input_shape=(50, 50, 1)),
        MaxPooling2D((2, 2), padding='valid'),
        Dropout(0.3),
        Conv2D(32, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Dropout(0.3),
        Conv2D(64, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Dropout(0.3),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_labels, activation='softmax'),
    ]


def model_2(num_labels):
    return [
        Conv2D(32, kernel_size=(3, 3), activation='relu', padding='valid', input_shape=(50, 50, 1)),
        MaxPooling2D((2, 2), padding='valid'),
        Dropout(0.25),
        Conv2D(64, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu', padding='valid'),
        MaxPooling2D(pool_size=(2, 2), padding='valid'),
        Dropout(0.4),

        Flatten(),
        Dense(2048, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.3),
        Dense(num_labels, activation='softmax'),
    ]
