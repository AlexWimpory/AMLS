import math
from collections import OrderedDict
import cv2
import dlib
from A2_face_recogniser import read_image, draw_rect_on_image, show_images, crop_faces
import numpy as np

"""This represents the point numbers returned in the shape object from dblib"""
FACIAL_LANDMARKS_IDXS = OrderedDict([
    ("mouth", (48, 68)),
    ("right_eyebrow", (17, 22)),
    ("left_eyebrow", (22, 27)),
    ("right_eye", (36, 42)),
    ("left_eye", (42, 48)),
    ("nose", (27, 35)),
    ("jaw", (0, 17))
])


class FeaturesDetector:
    """ Class used to detect facial outline using OpenCV Haas model"""

    def __init__(self):
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    def detect_features(self, image, rects):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return [self._predictor(gray, rect) for rect in rects]

    def detect_faces(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self._detector(gray, 1)


# A bunch of helper functions


def rectangle_to_numpy(rect):
    return np.array([rect.left(), rect.top(), rect.width(), rect.height()])


def shapes_to_coords(shapes, landmarks=None):
    coords = []
    for shape in shapes:
        if not landmarks:
            for i in range(0, shape.num_parts):
                coords.append((shape.part(i).x, shape.part(i).y))
        else:
            for landmark in landmarks:
                for i in range(landmark[0], landmark[1]):
                    coords.append((shape.part(i).x, shape.part(i).y))
    return coords


def draw_points_on_image(coords, image):
    new_new_image = image.copy()
    for coord in coords:
        cv2.circle(new_new_image, coord, 2, (255, 0, 0), -1)
    return new_new_image


def shapes_to_rects(shapes, landmarks):
    rects = []
    for shape in shapes:
        for landmark in landmarks:
            min_x = math.inf
            min_y = math.inf
            max_x = 0
            max_y = 0
            for i in range(landmark[0], landmark[1]):
                min_x = min(shape.part(i).x, min_x)
                min_y = min(shape.part(i).y, min_y)
                max_x = max(shape.part(i).x, max_x)
                max_y = max(shape.part(i).y, max_y)
            rects.append((min_x, min_y, max_x - min_x, max_y - min_y))
    return rects


def copy_rects_on_image(target_image, img, rcts):
    for rect in rcts:
        region = img[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        target_image[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]] = region
    return target_image


if __name__ == '__main__':
    img = read_image("../Datasets/celeba/img/7.jpg")
    # Here we are using a dlib detector which does both faces and features
    detector = FeaturesDetector()
    # This detects a set of faces and is needed in native form for the next step
    fcs = detector.detect_faces(img)
    # Cropping is more difficult because we have a dlib class
    crop_fcs = crop_faces(img, [rectangle_to_numpy(rect) for rect in fcs])
    # This takes the list of faces and detects the features in each face
    # Note a shape is a dlib object which contains a list of recognition points
    shps = detector.detect_features(img, fcs)
    # For each shape turn it into a set of coordinates
    crds = shapes_to_coords(shps)
    # Or we can pick out specific features
    # crds = shapes_to_coords(shps, [FACIAL_LANDMARKS_IDXS['left_eye'], FACIAL_LANDMARKS_IDXS['right_eye']])
    rcts = shapes_to_rects(shps, [
        FACIAL_LANDMARKS_IDXS['left_eye'], FACIAL_LANDMARKS_IDXS['right_eye'], FACIAL_LANDMARKS_IDXS['mouth']
    ])
    # Drawing from here
    # Draw the main image with the faces outlined
    new_image_face = draw_rect_on_image([rectangle_to_numpy(rect) for rect in fcs] + rcts, img)
    # Draw the cropped faces one at a time
    # Draw the main image with the features
    new_image_cords = draw_points_on_image(crds, img)
    # Draw just the features on a blank image
    blank_image = np.zeros(shape=img.shape, dtype=np.uint8)
    new_image_cords_only = draw_points_on_image(crds, blank_image)
    # Copy rects to the new blank image
    blank_image = np.zeros(shape=img.shape, dtype=np.uint8)
    new_image_rects_only = copy_rects_on_image(blank_image, img, rcts)
    # Show all the images
    show_images([new_image_face, new_image_cords, new_image_cords_only, new_image_rects_only], "face2.jpg")