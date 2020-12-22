from A2.A2_model_labeler import ModelLabelEncoder
from tensorflow.python.keras.models import load_model
from A2_feature_pre_processing import feature_pre_processor
from A2_model_plotter import plot_confusion_matrix
from tensorflow.python.ops.confusion_matrix import confusion_matrix
from numpy.ma import argmax
import numpy as np
import pandas as pd


"""
Allows the model to be tested on the data in the celeba_test repository, printing an accuracy and a confusion matrix
Also contains the function for calculating matrix required for the confusion matrix
"""


class ModelEvaluator:
    def __init__(self, model_name):
        self._model = load_model(f'{model_name}.hdf5')
        self._le = ModelLabelEncoder.load(model_name)

    def test_model(self, x_data, y_data):
        score = self._model.evaluate(x_data, y_data, verbose=0)
        accuracy = 100 * score[1]
        plot_confusion_matrix(calculate_confusion_matrix(self._model, self._le, x_data, y_data))
        return accuracy

    def evaluate(self):
        features_and_labels = feature_pre_processor('../Datasets/celeba_test')
        labels = features_and_labels['labels'].tolist()
        ftrs = np.array(features_and_labels['image_feature'].to_list())
        label_encoder = ModelLabelEncoder.load('A2')
        accuracy = self.test_model(ftrs, label_encoder.transform(labels))
        print(f'Testing accuracy = {accuracy:.4f}')


def calculate_confusion_matrix(model, le, x_test, y_test):
    y_pred = model.predict_classes(x_test)
    y_test = argmax(y_test, axis=1)
    con_mat = confusion_matrix(labels=y_test, predictions=y_pred).numpy()
    con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
    classes = le.inverse_transform([0, 1])
    return pd.DataFrame(con_mat_norm, index=classes, columns=classes)


def evaluate():
    ModelEvaluator('A2').evaluate()


if __name__ == '__main__':
    evaluate()
