from PIL import Image

import model
import preprocessing
import constants
import numpy as np


def main():
    face_recognizer = model.FaceRecognizer().load(constants.MODEL_PATH)
    image = Image.open(constants.INPUT_FILE)
    image = preprocessing.ImagePreprocessor().convert_image(image)

    if not image:
        result_file = open(constants.OUTPUT_FILE, "w")
        result_file.write("NOT FOUND")
        result_file.close()
        return

    image = np.array(image, 'uint8')

    image, y = preprocessing.ImagePreprocessor().filter_images([image], [1])

    probability = face_recognizer.predict_proba(image)[0]

    result_file = open(constants.OUTPUT_FILE, "w")

    if max(probability) >= 0.6:
        answer = probability.index(max(probability))
        result_file.write(face_recognizer.get_target_names()[answer] + "\n")
        result_file.write(str(max(probability)))

    else:
        result_file.write("NOT FOUND")

    result_file.close()


if __name__ == "__main__":
    main()
