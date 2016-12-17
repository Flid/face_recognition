from __future__ import unicode_literals, absolute_import

import dlib


class Visualizer(object):
    def __init__(self):
        self._win = dlib.image_window()

    def clear(self):
        self._win.clear_overlay()

    def show(self, img):
        self._win.set_image(img)
