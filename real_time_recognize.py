from __future__ import unicode_literals, absolute_import

import os
import logging
from PIL import Image
from face_recognition.capture import Camera
from face_recognition.detect import FaceDetector
from face_recognition.visualizer import Visualizer
from face_recognition.recognizer import Recognizer

log = logging.getLogger(__name__)

predictor = FaceDetector(
    b'/home/anton/tmp/shape_predictor_68_face_landmarks.dat',
)
camera = Camera()
vis = Visualizer()


def load_images():
    PATH = '/home/anton/hobby/data_sets/photos/'
    NAMES = ['anton', 'sveta']

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


recognizer = Recognizer()
recognizer.fit(load_images())


while True:
    img = camera.get_frame()
    vis.show(img)

    faces = predictor.find_faces(img)

    if not faces:
        print 'nothing'
        continue

    result = recognizer.recognize(faces)
    print result
