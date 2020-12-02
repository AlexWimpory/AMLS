from B2.B2_model_trainer import ModelLabelEncoder
from tensorflow.python.keras.models import load_model
from B2_feature_pre_processing import feature_pre_processor
import numpy as np


class ModelEvaluator:
    def __init__(self, model_name):
        self._model = load_model(f'{model_name}.hdf5')
        self._le = ModelLabelEncoder.load(model_name)

    def test_model(self, x_data, y_data):
        score = self._model.evaluate(x_data, y_data, verbose=0)
        accuracy = 100 * score[1]
        return accuracy

    def evaluate(self):
        features_and_labels = feature_pre_processor('../Datasets/cartoon_set_test')
        labels = features_and_labels['labels'].tolist()
        ftrs = np.array(features_and_labels['image_feature'].to_list())
        label_encoder = ModelLabelEncoder.load('B2')
        accuracy = self.test_model(ftrs, label_encoder.transform(labels))
        print(f'Testing accuracy = {accuracy:.4f}')


def evaluate():
    ModelEvaluator('B2').evaluate()


if __name__ == '__main__':
    evaluate()
