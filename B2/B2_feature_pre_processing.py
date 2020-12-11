from pandas import DataFrame
from B2_feature_extractor import image_to_array, create_image
from B2_file_utils import return_from_path
from functools import partial
import os
import pickle
from B2_ground_truth_processor import GroundtruthReader


def prepare_feature(ground_truth, file_name):
    """Generate features and look up the labels"""
    base_file_name = os.path.basename(file_name)
    gtp = GroundtruthReader(ground_truth)
    from_ground_truth = gtp.lookup_filename(base_file_name)
    image_feature = image_to_array(create_image(file_name))
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
                            '.png')
    return DataFrame(ftrs)


if __name__ == '__main__':
    save_features(feature_pre_processor('../Datasets/cartoon_set'), 'B2.data')
