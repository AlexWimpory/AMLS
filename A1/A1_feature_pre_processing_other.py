from pandas import DataFrame
from A1_feature_extractor import image_to_vector, create_image
from A1_file_utils import return_from_path
from functools import partial
import os
import pickle
from A1_ground_truth_processor import GroundtruthReader


def prepare_feature(ground_truth, file_name):
    """Generate features and look up the labels"""
    base_file_name = os.path.basename(file_name)
    gtp = GroundtruthReader(ground_truth)
    from_ground_truth = gtp.lookup_filename(base_file_name)
    format_image = create_image(file_name)
    image_feature = image_to_vector(format_image)
    return {'image_feature': image_feature, 'labels': from_ground_truth}


def save_features(features, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(features, file)


def load_features(file_name):
    with open(file_name, 'rb') as file:
        return pickle.load(file)


def feature_pre_processor(path):
    prepare_feature_groundtruth = partial(prepare_feature, f'{path}/labels.csv')
    ftrs = return_from_path(prepare_feature_groundtruth,
                            f'{path}/img',
                            '.jpg')
    return DataFrame(ftrs)


if __name__ == '__main__':
    save_features(feature_pre_processor('../Datasets/celeba'), 'A1_SVM.data')



