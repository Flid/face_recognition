from __future__ import unicode_literals, absolute_import

import cv2


class Camera(object):
    def __init__(self):
        self._cap = cv2.VideoCapture(0)

    def get_frame(self):
        ret, img = self._cap.read()
        return img

    def __del__(self):
        self._cap.release()
