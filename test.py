import heapq
import os
import urllib

import cv2
import joblib
import numpy
from keras.engine.saving import load_model
from scipy.misc import imread, imresize

from helpers import get_im, HEIGHT, WIDTH


model = load_model("model.h5",)
lb = joblib.load("lb.sav")
print(lb.classes_)
while 1:
    try:
        print("\nРасположение изображения:")
        inp = "test/"+input()

        x = imread(inp, mode='L')
        x = imresize(x, (WIDTH, HEIGHT))

        x = x.reshape(1, WIDTH, HEIGHT, 1) / 255

        print(x.shape)
        pred = model.predict(x)[0]

        map_output = {}
        for i in range(len(pred)):
            print(lb.classes_[i], "\t", pred[i])

        max_pred = heapq.nlargest(1, range(len(pred)), pred.take)[0]
        print("Результат:", lb.classes_[max_pred])
    except Exception as e:
        print(repr(e))