from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping
from A1.A1_model_labeler import ModelLabelEncoder
from A1.A1_model_structures import *
from A1.A1_model_plotter import plot_history, plot_confusion_matrix
import time
import numpy as np
from A1 import A1_features_config as config
from A1_file_utils import load_object, save_object
from A1_model_evaluator import calculate_confusion_matrix

"""
* Load the pre-processed data
* Builds the model using the structure defined in model_structures
* Compiles the model
* Split the data into train, test and validate
* Train the model
* Plot learning curves
* Calculate accuracy on each data set
* Plot confusion matrix
"""


class ImageFeaturesModel:
    def __init__(self, model_name, le, layers):
        self.le = le
        self.model = Sequential(name=model_name)

        for layer in layers:
            self.model.add(layer)

    def compile(self):
        self.model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=config.optimizer)
        self.model.summary()

    def test_model(self, x_data, y_data):
        score = self.model.evaluate(x_data, y_data, verbose=0)
        accuracy = 100 * score[1]
        return accuracy

    def train_model(self, x_train, y_train, x_val, y_val):
        early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        checkpointer = ModelCheckpoint(filepath=f'{self.model.name}.hdf5', verbose=1, save_best_only=True)
        history = self.model.fit(x_train, y_train, batch_size=config.num_batch_size,
                                 epochs=config.num_epochs, validation_data=(x_val, y_val),
                                 callbacks=[early_stop, checkpointer], verbose=1)
        self.le.save(self.model.name)
        return history


def train_and_test_model(features, le, model):
    print(features.shape)
    x_train, x_test, y_train, y_test = train_test_split(features, le.encoded_labels,
                                                        test_size=1 - config.train_ratio,
                                                        random_state=44)
    x_cv, x_test, y_cv, y_test = train_test_split(x_test, y_test,
                                                  test_size=config.test_ratio / (
                                                          config.test_ratio + config.validation_ratio),
                                                  random_state=44)

    pre_acc = model.test_model(x_test, y_test)
    print(f'Pre-trained accuracy = {pre_acc:.4f}')

    history = model.train_model(x_train, y_train, x_cv, y_cv)
    plot_history(history)
    save_object(history.history, f'history_{config.optimizer}_{config.num_batch_size}.data')

    post_acc_train = model.test_model(x_train, y_train)
    print(f'Training accuracy = {post_acc_train:.4f}')

    post_acc_cv = model.test_model(x_cv, y_cv)
    print(f'Cross-validation accuracy = {post_acc_cv:.4f}')

    post_acc_test = model.test_model(x_test, y_test)
    print(f'Testing accuracy = {post_acc_test:.4f}')

    plot_confusion_matrix(calculate_confusion_matrix(model.model, le, x_test, y_test))

    return post_acc_test


def trainer():
    features_and_labels = load_object('A1.data')
    labels = features_and_labels['labels'].tolist()
    ftrs = np.array(features_and_labels['image_feature'].to_list())
    label_encoder = ModelLabelEncoder(labels)
    mdl_structure = model_1(label_encoder.encoded_labels.shape[1])
    mdl = ImageFeaturesModel('A1', label_encoder, mdl_structure)
    mdl.compile()
    return train_and_test_model(ftrs, label_encoder, mdl)


if __name__ == '__main__':
    parameters = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    results = []
    execution_time = []
    for batch_size in parameters:
        config.optimizer = 'adam'
        config.num_batch_size = batch_size
        start = time.time()
        results.append(trainer())
        end = time.time()
        execution_time.append(end - start)

    print(results)
    print(execution_time)
