import glob
import numpy as np
import cv2
from A1_face_recogniser import FaceDetector, crop_faces

"""
Converts an image to data which is usable by the machine learning model
Face detection is commented out as I found that it did not make a difference to the output
"""

#detector = FaceDetector()

def create_image(path):
    """Create an image from the passed path"""
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # faces = detector.detect_faces(image)
    # if len(faces) > 0:
    #     image = crop_faces(image, faces)[0]
    image = cv2.resize(image, (50, 50))
    return image


def image_to_array(image):
    """Turn the image into an array of numbers the same size as the image,
     between 0 and 255 depending on the pixel intensity (used for NN) """
    image_array = np.array(image).astype('float32')
    image_array = image_array / 255
    image_array = image_array.reshape(50, 50, 1)
    return image_array


def image_to_vector(image):
    """Some models can't take an array as an input so the array is flattened to a vector (used for SVM and  KNN)"""
    image_vector = np.array(image).reshape(-1)
    image_vector = image_vector / 255
    return image_vector


def visualise(image, title):
    """Visualise processed image"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    feature = create_image('../Datasets/celeba/img/4512.jpg')
    # data = image_to_vector(feature)
    data = image_to_array(feature)
    visualise(feature, 'Image')
