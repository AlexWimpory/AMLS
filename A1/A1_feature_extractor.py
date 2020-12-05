import glob

import numpy as np
import cv2

from A1_face_recogniser import FaceDetector, crop_faces

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


def image_to_data(image):
    image_array = np.array(image).astype('float32')
    image_array = image_array / 255
    image_array = image_array.reshape(50, 50, 1)
    return image_array


def image_to_vector(image):
    return np.array(image).reshape(-1)


def visualise(image, title):
    """Visualise processed image"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    feature = create_image('../Datasets/celeba/img/118.jpg')
    data = image_to_vector(feature)
    # data = image_to_data(feature)
    # # visualise(feature, 'Image')
    # path = glob.glob("../Datasets/celeba/img/*.jpg")
    # X = []
    # count = 0
    # for img in path:
    #         n = cv2.imread(img)
    #         gray = cv2.cvtColor(n, cv2.COLOR_BGR2GRAY)
    #         gray = cv2.resize(gray, (50, 50))
    #         X.append(gray)
    #         count = count + 1
    # sizeImg = X[0].shape
    # A = np.zeros((sizeImg[0] * sizeImg[1], len(X)))
    #
    # for i in range(0, len(X)):
    #     tmp = (np.array(X[i]).reshape(-1))
    #     A[:, i] = np.array(tmp)
    # print(A.shape)
