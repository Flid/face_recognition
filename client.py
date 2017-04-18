from __future__ import absolute_import

import json
from StringIO import StringIO
from PIL import Image
import requests
from face_recognition.conf import RECOGNIZER_URL


def send(img):
    buf = StringIO()
    img.save(buf, format='JPEG', quality=90)
    buf.seek(0)

    response = requests.post(
        RECOGNIZER_URL + '/recognize',
        files={'img': buf},
    )

    if response.status_code != 200:
        return None

    return json.loads(response.content)


if __name__ == '__main__':
    img = Image.open(open('/home/anton/tmp/cam2.bmp', 'rb'))
    print send(img)
