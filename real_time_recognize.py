from __future__ import unicode_literals, absolute_import

import os
import logging
from PIL import Image
from face_recognition.capture import Camera
from face_recognition.detect import FaceDetector
from face_recognition.visualizer import Visualizer
from face_recognition.recognizer import Recognizer
import pickle


log = logging.getLogger(__name__)

predictor = FaceDetector(
    b'shape_predictor_68_face_landmarks.dat',
)
cached_recognizer_path = 'recognizer.dump'

camera = Camera()
vis = Visualizer()


def load_images():
    PATH = '/home/anton/hobby/data_sets/photos/'
    NAMES = ['anton', 'sveta', 'ian']

    output = {}

    for i, name in enumerate(NAMES):
        lpath = os.path.join(PATH, name)
        output[name] = []

        for fname in os.listdir(lpath):
            if not fname.lower().endswith('.jpg'):
                continue
            img = Image.open(os.path.join(lpath, fname)).convert('L')
            output[name].append(img)

    return output


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

print('Starting main loop...')

while True:
    img = camera.get_frame()
    vis.show(img)

    faces = predictor.find_faces(img)

    if not faces:
        print 'nothing'
        continue

    result = recognizer.recognize(faces)
    print result
