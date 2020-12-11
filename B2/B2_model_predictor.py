import json
from B2.B2_feature_extractor import create_image, image_to_array
from tensorflow.python.keras.models import load_model
from B2.B2_model_trainer import ModelLabelEncoder
import numpy as np


class ModelPredictor:
    def __init__(self, model_name):
        self._model = load_model(f'{model_name}.hdf5')
        self._le = ModelLabelEncoder.load(model_name)

    def predict(self, file_name):
        results = ModelPredictorResults()
        image = np.array([image_to_array(create_image(file_name))])
        predicted_vector = self._model.predict_classes(image)
        predicted_class = self._le.inverse_transform(predicted_vector)
        results.predicted_class = predicted_class[0]

        predicted_probability_vector = self._model.predict(image)
        predicted_probability = predicted_probability_vector[0]

        for i in range(len(predicted_probability)):
            category = self._le.inverse_transform(np.array([i]))
            results.predicted_probabilities[category[0]] = str(format(predicted_probability[i], '.8f'))

        return results


class ModelPredictorResults:
    def __init__(self):
        self.predicted_class = None
        self.predicted_probabilities = {}


def predict():
    predictor = ModelPredictor(model_name='B2')
    res = predictor.predict('../Datasets/cartoon_set/img/10.png')
    print(json.dumps(res.__dict__))


if __name__ == '__main__':
    predict()
