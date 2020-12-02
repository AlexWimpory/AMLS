import numpy as np
import cv2


def create_image(path):
    """Create an image from the passed path"""
    image = cv2.imread(path)
    image = cv2.resize(image, (50, 50))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def image_to_data(image):
    image_array = np.array(image).astype('float32')
    image_array = image_array / 255
    image_array = image_array.reshape(50, 50, 1)
    return image_array


def visualise(image, title):
    """Visualise processed image"""
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    feature = create_image('../Datasets/cartoon_set/img/5.png')
    data = image_to_data(feature)
    visualise(feature, 'Image')
