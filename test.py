import os
import urllib

import cv2
import joblib
import numpy
from keras.engine.saving import load_model

from helpers import get_im, HEIGHT, WIDTH


model = load_model("model.h5",)
lb = joblib.load("mlb.sav")
print(lb.classes_)
while 1:
    print("Путь до файла: ")
    inp = input()
    dataX = []
    binary_image = get_im(inp)
    dataX.append(binary_image)
    dataX = numpy.array(dataX)/255
    dataX = dataX.reshape((-1, HEIGHT, WIDTH, 1))
    print(dataX.shape)
    pred = list(model.predict(dataX)[0])
    for i in range(len(pred)):
        print(lb.classes_[i],pred[i])