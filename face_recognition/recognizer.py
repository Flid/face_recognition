from __future__ import unicode_literals, absolute_import

import numpy as np
import logging

from PIL import Image
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import SVC


log = logging.getLogger(__name__)


class Recognizer:
    SIZE = 96
    N_COMPONENTS = 100

    def __init__(self):
        self.labels = None
        self._clf = None
        self._pca = None

    def fit(self, photos_dict):
        """
        {'name': [img1, img2]}
        """

        X_raw = []
        Y = []
        self.labels = []

        i = 0
        for key, photos in photos_dict.iteritems():
            for img in photos:
                img = img.resize((self.SIZE, self.SIZE))
                X_raw.append(img)
                Y.append(i)

            self.labels.append(key)
            i += 1

        X = np.ndarray(shape=(len(X_raw), self.SIZE*self.SIZE))

        for i in range(len(X_raw)):
            X[i] = np.array(X_raw[i]).reshape(self.SIZE*self.SIZE)

        self._pca = PCA(
            n_components=self.N_COMPONENTS,
            svd_solver='randomized',
            whiten=True,
        )
        self._pca.fit(X)

        #eigenfaces = pca.components_.reshape((self.N_COMPONENTS, self.SIZE, self.SIZE))

        X_pca = self._pca.transform(X)

        # Train a SVM classification model
        param_grid = {
            b'C': [1e3, 5e3, 1e4, 5e4, 1e5],
            b'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
        }

        clf = GridSearchCV(
            SVC(kernel=b'rbf', class_weight=b'balanced', probability=True),
            param_grid,
        )
        self._clf = clf.fit(X_pca, Y)

    def recognize(self, images):
        processed_images = []
        for img in images:
            img = Image.fromarray(img).resize((self.SIZE, self.SIZE))
            img = np.array(img).reshape(self.SIZE*self.SIZE)
            processed_images.append(img)

        img_pca = self._pca.transform(processed_images)

        probabilities = self._clf.predict_proba(img_pca)

        output = []

        for probs in probabilities:
            best_fit = max(enumerate(probs), key=lambda item: item[1])
            output.append((
                self.labels[best_fit[0]],
                best_fit[1],
            ))

        return output
