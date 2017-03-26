import os

from PIL import Image
import numpy as np
import constants
import preprocessing
import model


def main():
    image_preprocessor = preprocessing.ImagePreprocessor()

    X_load = []
    Y_load = []

    classes = dict()

    for path in os.listdir(constants.PHOTO_PATH):
        classes[path] = 0
        for file_name in os.listdir(constants.PHOTO_PATH + "//" + path):
            file_path = constants.PHOTO_PATH + "//" + path + "//" + file_name
            image = image_preprocessor.convert_image(Image.open(file_path))
            if not image:
                continue

            image = np.array(image, 'uint8')
            classes[path] += 1
            X_load.append(image)
            Y_load.append(path)

    X = []
    Y = []

    for x, y in zip(X_load, Y_load):
        if classes[y] < constants.MINIMUM_PHOTOS:
            continue

        X.append(x)
        Y.append(y)

    X, Y = image_preprocessor.filter_images(X, Y)

    face_recognizer = model.FaceRecognizer()
    face_recognizer.fit(X, Y)
    face_recognizer.save(constants.MODEL_PATH)


if __name__ == "__main__":
    main()
