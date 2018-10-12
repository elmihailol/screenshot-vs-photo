import os
import urllib

import cv2
import joblib
import numpy
from keras.engine.saving import load_model

HEIGHT = 128
WIDTH = 128


def get_im(path):
    # Загружаем изображение
    img = cv2.imread(path, 0)
    # Конвертируем изображение
    resized = cv2.resize(img, (HEIGHT, WIDTH))
    return resized


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = load_model("model.h5")
lb = joblib.load("lb.sav")
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