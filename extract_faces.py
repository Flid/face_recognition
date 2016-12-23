from __future__ import unicode_literals, absolute_import

from PIL import Image
from face_recognition.capture import Camera
from face_recognition.detect import FaceDetector
from face_recognition.visualizer import Visualizer


predictor = FaceDetector(
    b'/home/anton/tmp/shape_predictor_68_face_landmarks.dat',
)
camera = Camera()
vis = Visualizer()


N = 0

while True:
    img = camera.get_frame()

    faces = predictor.find_faces(img)

    if len(faces) != 1:
        continue

    vis.show(faces[0])

    N += 1
    img = Image.fromarray(faces[0])


    #with open('/tmp/out/%s.jpg' % N, 'w') as fd:
    #    img.save(fd)
