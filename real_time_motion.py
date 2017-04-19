import logging
from face_recognition.capture import Camera
from face_recognition.motion import MotionDetector
from face_recognition.visualizer import Visualizer
from face_recognition.utils import get_recognizer_cached
from PIL import Image
import numpy as np

log = logging.getLogger(__name__)

detector = MotionDetector(min_area=1000)

camera = Camera()
vis = Visualizer()

recognizer = get_recognizer_cached()

print('Starting main loop...')

while True:
    img = camera.get_frame()
    img2 = Image.fromarray(img)
    img = img2.resize((640, 480))
    img = np.array(img)

    vis.show(img)

    motions = detector.submit_frame(img)
    print motions
