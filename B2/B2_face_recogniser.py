import cv2


class FaceDetector:
    """ Class used to detect facial outline using OpenCV Haas model"""

    def __init__(self):
        self._face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_faces(self, image):
        # Detect the faces
        return self._face_cascade.detectMultiScale(image, 1.1, 4)

# A bunch of helper functions


def crop_faces(image, rectangle_list, return_original=False):
    """Crop an image which is really just a matter of reducing a numpy array"""
    faces = []
    for rect in rectangle_list:
        coords = convert_coords_to_points(rect)
        faces.append(image[coords[0][1]:coords[1][1], coords[0][0]:coords[1][0]])
    if return_original and len(faces) == 0:
        return [image]
    return faces


def draw_rect_on_image(shape_list, image):
    """Draw rectangles on an image but return a new image"""
    copy_image = image.copy()
    for shape in shape_list:
        # Draw a rectangle on the original image
        coords = convert_coords_to_points(shape)
        cv2.rectangle(img=copy_image,
                      pt1=coords[0],
                      pt2=coords[1],
                      color=(0, 255, 0),
                      thickness=4)
    return copy_image


def convert_coords_to_points(coords):
    """Convert the numpy array to a tuple of coordinates"""
    return (coords[0], coords[1]), (coords[0] + coords[2], coords[1] + coords[3])


def read_image(path):
    # read the image
    image = cv2.imread(path)
    # pre process the image
    return image


def show_images(images, window_name=None):
    for image in zip(range(len(images)), images):
        cv2.imshow(mat=image[1], winname=window_name + str(image[0]))
    # Wait for a key press to exit
    cv2.waitKey()
    # Close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    img = read_image("../Datasets/cartoon_set/img/4.png")
    # This is what you do to return a cropped to face image
    # You need a face detector which includes a model so its a class - create this only once
    detector = FaceDetector()
    # This returns a list of faces found in the image
    fcs = detector.detect_faces(img)
    # This takes an image and returns a list of images cropped to the faces
    # Note that if you want to return the image even if no face is detected there is a parameter you can set
    crop_fcs = crop_faces(img, fcs)
    # Just drawing stuff from here
    new_image = draw_rect_on_image(fcs, img)
    show_images([new_image] + crop_fcs, "face2.jpg")