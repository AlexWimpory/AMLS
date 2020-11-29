import json
from A1.feature_extractor import create_mfcc, mfcc_mean
from tensorflow.python.keras.models import load_model
from A1.model_trainer import ModelLabelEncoder
import numpy as np


class ModelPredictor:
    def __init__(self, model_name):
        self._model = load_model(f'data/{model_name}.hdf5')
        self._le = ModelLabelEncoder.load(model_name)

    def predict(self, file_name):
        results = ModelPredictorResults()
        mfcc = np.array([mfcc_mean(create_mfcc(file_name))])
        predicted_vector = self._model.predict_classes(mfcc)
        predicted_class = self._le.inverse_transform(predicted_vector)
        results.predicted_class = predicted_class[0]

        predicted_probability_vector = self._model.predict(mfcc)
        predicted_probability = predicted_probability_vector[0]

        for i in range(len(predicted_probability)):
            category = self._le.inverse_transform(np.array([i]))
            results.predicted_probabilities[category[0]] = str(format(predicted_probability[i], '.8f'))

        return results


class ModelPredictorResults:
    def __init__(self):
        self.predicted_class = None
        self.predicted_probabilities = {}


if __name__ == '__main__':
    predictor = ModelPredictor(model_name='model_1')
    res = predictor.predict('D:\\Audio Features\\UrbanSound8K\\UrbanSound8K\\audio\\fold10\\2937-1-0-0.wav')
    print(json.dumps(res.__dict__))
