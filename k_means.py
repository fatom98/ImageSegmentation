import cv2 as cv
import numpy as np

from image import Image


class KMeans:
    def __init__(self, image: Image, shape):
        self.__img = image
        self.__shape = shape
        self.__clustered_image: np.ndarray = None
        self.__history = []

    def run(self, params: list[int]) -> int:
        k = round(params[0])
        iteration = round(params[1])
        attempts = round(params[2])
        epsilon = params[3]

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, iteration, epsilon)

        compactness, label, center = cv.kmeans(self.__img.array, k, None, criteria, attempts, cv.KMEANS_RANDOM_CENTERS)
        self.__history.append(compactness)

        center = np.uint8(center)

        res = center[label.flatten()]
        self.__clustered_image = res.reshape(self.__shape)

        return int(compactness)

    @property
    def clustered_image(self):
        return self.__clustered_image

    @property
    def history(self):
        return self.__history
