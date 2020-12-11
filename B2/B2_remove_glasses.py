import os
from functools import partial
from B2_eye_recogniser import FeaturesDetector
from B2_face_recogniser import read_image
from B2_file_utils import apply_to_path

detector = FeaturesDetector()


def remove_glasses(dtc, path):
    img = read_image(path)
    fcs = dtc.detect_faces(img)
    shps = detector.detect_features(img, fcs)
    if len(shps) == 0:
        os.rename(path, path.replace('img', 'img_glasses'))


if __name__ == '__main__':
    partial_remove_glasses = partial(remove_glasses, detector)
    apply_to_path(partial_remove_glasses, '../Datasets/cartoon_set/img', '.png')
