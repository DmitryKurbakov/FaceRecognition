import cv2
import numpy as np
from PIL.Image import fromarray
import constants

class ImagePreprocessor:
    def __init__(self):
        self._faceCascade = cv2.CascadeClassifier(constants.CURRENT_DIRECTORY + "/haarcascade_frontalface_default.xml")

    def convert_image(self, image):
        image = image.convert('L')
        image = np.array(image, 'uint8')
        faces = self._faceCascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            return fromarray(image[y: y + h, x: x + w]).resize((50, 50))

        return None

    def filter_images(self, X, Y):
        X_result = []
        Y_result = []

        for x, y in zip(X, Y):
            x_row = []

            for row in x:
                x_row = x_row + row.tolist()

            X_result.append(x_row)
            Y_result.append(y)

        return X_result, Y_result
