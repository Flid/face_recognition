import os
from PIL import Image
import pickle
from face_recognition.recognizer import Recognizer
from face_recognition.conf import RECOGNIZER_TRAINING_NAMES, RECOGNIZER_TRAINING_SET_DIR


def load_images():
    output = {}

    for i, name in enumerate(RECOGNIZER_TRAINING_NAMES):
        lpath = os.path.join(RECOGNIZER_TRAINING_SET_DIR, name)
        output[name] = []

        for fname in os.listdir(lpath):
            if not fname.lower().endswith('.jpg'):
                continue

            img = Image.open(os.path.join(lpath, fname)).convert('L')
            output[name].append(img)

    return output


def get_recognizer_cached(cached_recognizer_path='recognizer.dump'):
    try:
        with open(cached_recognizer_path, 'r') as fd:
            print('Loading cached Serializer...')
            recognizer = pickle.load(fd)
    except IOError:
        print('Educating Recognizer...')
        recognizer = Recognizer()
        recognizer.fit(load_images())

        with open(cached_recognizer_path, 'w') as fd:
            print('Caching Serializer...')
            pickle.dump(recognizer, fd)

    return recognizer
