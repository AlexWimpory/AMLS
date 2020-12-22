from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
import pickle

"""
Build and save an encoder that changes labels type
* Word labels = [word_1,word_2,word_3]
* Numerical labels = [0,1,2]
* One hot encoding labels = [1,0,0],[0,1,0],[0,0,1]
"""


class ModelLabelEncoder:
    def __init__(self, labels):
        self._le = LabelEncoder()
        label_array = np.array(labels).ravel()
        self.encoded_labels = to_categorical(self._le.fit_transform(label_array))

    def transform(self, labels):
        return to_categorical(self._le.transform(labels))

    def inverse_transform(self, data):
        return self._le.inverse_transform(data)

    def save(self, model_name):
        with open(f'{model_name}_labels.data', 'wb') as fout:
            pickle.dump(self, fout)

    @staticmethod
    def load(model_name):
        with open(f'{model_name}_labels.data', 'rb') as fin:
            return pickle.load(fin)
