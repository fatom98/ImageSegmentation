from typing import Self

import cv2 as cv
import numpy as np


class Image:
    def __init__(self, *, image: np.ndarray = None, path: str = None):
        self.__img: np.ndarray = cv.imread(path) if path is not None else image

    def resize(self, width: int, height: int) -> Self:
        new_img = cv.resize(self.__img, (width, height), interpolation=cv.INTER_AREA)
        return Image(image=new_img)

    def reshape(self, shape: tuple[int, int]) -> Self:
        new_img = self.__img.reshape(shape)
        return Image(image=new_img)

    def to_float_32(self) -> Self:
        new_img = np.float32(self.__img)
        return Image(image=new_img)

    @property
    def shape(self):
        return self.__img.shape

    @property
    def array(self):
        return self.__img
