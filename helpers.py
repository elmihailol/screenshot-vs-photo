import os

import cv2
from scipy.misc import imresize
from scipy.misc import imread

from config import HEIGHT, WIDTH


def get_im(path):
    x = imread(path, mode='L')
    x = imresize(x, (HEIGHT, WIDTH))
    return x.reshape(1, HEIGHT, WIDTH, 1) / 255
