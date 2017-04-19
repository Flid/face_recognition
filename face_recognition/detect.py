from __future__ import unicode_literals, absolute_import

import logging
import numpy as np
import cv2
import dlib
from skimage import draw
import scipy.ndimage
import math
import os


log = logging.getLogger(__name__)


class FaceDetector(object):
    def __init__(self, debug=False):
        shape_predictor_path = os.path.join(
            os.path.dirname(__import__('face_recognition').__file__),
            'shape_predictor_68_face_landmarks.dat',
        )

        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(shape_predictor_path)
        self._debug = debug

    @staticmethod
    def _average(shape, start, end):
        points = [shape.part(i) for i in range(start, end + 1)]

        x = sum(p.x for p in points) / len(points)
        y = sum(p.y for p in points) / len(points)
        return x, y

    @staticmethod
    def _add_point(img, point, color=255):
        rr, cc = draw.circle(point[1], point[0], 3, img.shape)
        img[rr, cc] = color

    @staticmethod
    def transform_image(image, l_eye, r_eye, mouth, nose):
        orig_height = image.shape[0]
        orig_width = image.shape[1]

        angle1 = math.acos(
            (mouth[1] - nose[1]) / math.sqrt((nose[0]-mouth[0])**2 + (nose[1]-mouth[1])**2),
        )

        if nose[0] < mouth[0]:
            angle1 *= -1

        angle2 = math.acos(
            (r_eye[1] - l_eye[1]) / math.sqrt((r_eye[0]-l_eye[0])**2 + (r_eye[1]-l_eye[1])**2),
        )
        angle2 = math.pi / 2 - angle2

        angle = (angle1 + angle2) / 2.

        image = scipy.ndimage.interpolation.rotate(image, math.degrees(angle))

        angle *= -1

        eye_x = (l_eye[0] + r_eye[0]) / 2
        eye_y = (l_eye[1] + r_eye[1]) / 2

        # Convert (eye_x, eye_y) to new rotated coordinates
        mid_x = (eye_x-orig_width/2) * math.cos(angle) - (eye_y-orig_height/2) * math.sin(angle) + image.shape[1] / 2
        mid_y = (eye_x-orig_width/2) * math.sin(angle) + (eye_y-orig_height/2) * math.cos(angle) + image.shape[0] / 2

        face_size = math.sqrt(
            ((r_eye[0] + l_eye[0])/2 - mouth[0])**2
            + ((r_eye[1] + l_eye[1])/2 - mouth[1])**2
        ) * 3

        image = image[
            mid_y-face_size/2*0.9: mid_y+face_size/2*1.1,
            mid_x-face_size/2: mid_x+face_size/2,
        ]
        return image

    def find_faces(self, img):
        if len(img.shape) == 3 and img.shape[2] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Ask the detector to find the bounding boxes of each face. The 1 in the
        # second argument indicates that we should upsample the image 1 time. This
        # will make everything bigger and allow us to detect more faces.
        dets = self._detector(img, 1)
        log.debug('Number of faces detected: {}'.format(len(dets)))

        output = []

        for d in dets:
            # Get the landmarks/parts for the face in box d.
            shape = self._predictor(img, d)

            left_eye = self._average(shape, 37, 40)
            right_eye = self._average(shape, 43, 47)
            mouth = self._average(shape, 49, 67)
            nose = self._average(shape, 30, 35)

            if self._debug:
                # Draw the face landmarks on the screen.
                self._add_point(img, left_eye)
                self._add_point(img, right_eye)
                self._add_point(img, mouth)
                self._add_point(img, nose)

            img_copy = img[:]
            img_copy = self.transform_image(
                img_copy,
                left_eye,
                right_eye,
                mouth,
                nose,
            )

            if img_copy.shape[0] < 0 or img_copy.shape[1] < 10:
                continue

            img_copy = img_copy.astype(np.uint8)
            output.append(img_copy)

        return output
