from flask import Flask, request, jsonify
from face_recognition.utils import get_recognizer_cached
from face_recognition.detect import FaceDetector
from PIL import Image
import numpy as np
from time import time


def run():
    app = Flask(__name__)

    recognizer = get_recognizer_cached()
    predictor = FaceDetector(
        b'shape_predictor_68_face_landmarks.dat',
    )

    @app.route('/recognize', methods=['POST'])
    def process_image():
        raw_img = request.files.get('img')

        if not raw_img:
            return jsonify({'error': 'File not submitted'}, status=400)

        img = Image.open(raw_img)
        img_data = np.array(img)

        faces = predictor.find_faces(img_data)

        if not faces:
            return jsonify([])

        result = recognizer.recognize(faces)
        return jsonify(result)

    app.run(host='0.0.0.0', port=10010)


if __name__ == '__main__':
    run()
