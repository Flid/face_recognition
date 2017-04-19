from __future__ import unicode_literals, absolute_import

import logging
from face_recognition.capture import Camera
from face_recognition.detect import FaceDetector
from face_recognition.visualizer import Visualizer
from face_recognition.utils import get_recognizer_cached


log = logging.getLogger(__name__)

predictor = FaceDetector()

camera = Camera()
vis = Visualizer()

recognizer = get_recognizer_cached()

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
