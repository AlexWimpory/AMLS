from sklearn.model_selection import train_test_split
from A1.A1_feature_pre_processing import load_features
from A1.A1_model_labeler import ModelLabelEncoder
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


class ImageFeaturesModel:
    def __init__(self, model_name, le):
        self.le = le
        self.model_name = model_name
        self._model = SVC(kernel='linear', random_state=0)

    def test_model(self, x_data, y_data):
        y_pred = self._model.predict(x_data)
        y_test = y_data.tolist()
        y_test_np = np.asarray(y_test)
        disc = y_test_np - y_pred
        count = 0
        for i in disc:
            if i == 0:
                count += 1
        return (100 * count) / len(y_pred)

    def train_model(self, x_train, y_train):
        self._model.fit(x_train, y_train)

    def predict(self, x_test):
        return self._model.predict(x_test)


def train_and_test_model(features, le, mdl):
    print(features.shape)
    x_train, x_test, y_train, y_test = train_test_split(features, le.numerical_labels,
                                                        test_size=0.15,
                                                        random_state=44)

    sc = StandardScaler()
    x_train = sc.fit_transform(x_train)
    x_test = sc.transform(x_test)

    mdl.train_model(x_train, y_train)

    post_acc_test = mdl.test_model(x_test, y_test)
    print(f'Testing accuracy = {post_acc_test:.4f}')


def trainer():
    features_and_labels = load_features('A1_SVM.data')
    labels = features_and_labels['labels'].tolist()
    ftrs = np.array(features_and_labels['image_feature'].to_list())
    label_encoder = ModelLabelEncoder(labels)
    mdl = ImageFeaturesModel('A1_SVM', label_encoder)
    train_and_test_model(ftrs, label_encoder, mdl)


if __name__ == '__main__':
    trainer()
