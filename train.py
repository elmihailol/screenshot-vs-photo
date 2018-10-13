import configparser
import random

import os

from models import image_net_model, conv2d_model

#os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import joblib
import numpy
import os
import sys
import tensorflow as tf

from keras import Sequential, Input, Model
from keras.applications import InceptionV3, VGG16, MobileNetV2
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Flatten, GlobalAveragePooling2D
from keras_preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from config import IMAGES_PATH, MAXIMUM_IMAGES_PER_CLASS, HEIGHT, WIDTH, TRAIN_MODE
from helpers import get_im



class ImageContainer:
    def __init__(self, data_path):
        """
        Создает контейнер с изображениями
        :param data_path: dict {class1: path1, class2: path2, ....}
        """
        self.images = {}
        for key, val in data_path.items():
            counter = 0
            self.images[key] = []
            print(key, val)
            folder_list = os.listdir(val)
            for file in folder_list:
                if counter >= MAXIMUM_IMAGES_PER_CLASS:
                    break
                try:
                    binary_image = get_im(val + "/" + file)
                    self.images[key].append(binary_image[0])
                    counter += 1
                except Exception as e:
                    print(repr(e))

        for key, val in self.images.items():
            print(key, len(val))

    def get_data_images(self):
        dataX = []
        dataY = []
        for key, val in self.images.items():
            print(key, len(val))
            dataX.extend(val)
            dataY.extend([key]*len(val))

        c = list(zip(dataX, dataY))
        random.shuffle(c)
        dataX, dataY = zip(*c)

        dataX = numpy.array(dataX)
        lb = LabelBinarizer()
        lb.fit(dataY)
        lb = joblib.load("lb.sav")
        dataY = lb.transform(dataY)
        dataY = numpy.array(dataY)
        print(dataX.shape)
        return dataX, dataY



def main(argv=None, model="default"):
    print("Read Config")
    ic = ImageContainer(IMAGES_PATH)
    dataX, dataY = ic.get_data_images()
    print("dataX:", len(dataX))
    print("dataY:", len(dataY))
    if model == "default":
        model = conv2d_model(output_len=3)
    if model == "image_net_model":
        model = image_net_model()
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.3)

    datagen = ImageDataGenerator(
        featurewise_center=False,
        samplewise_center=False,
        featurewise_std_normalization=False,
        samplewise_std_normalization=False,
        zca_whitening=False,
        rotation_range=90,
        zoom_range=0.4,
        width_shift_range=0.4,
        height_shift_range=0.4,
        horizontal_flip=False,
        vertical_flip=False)

    print("start")
    max_acc = 0
    for i in range(100):
        if TRAIN_MODE == 1:
            model.fit_generator(datagen.flow(x_train, y_train, batch_size=16),
                                    epochs=1, steps_per_epoch=512,
                                    verbose=1)
        else:
            model.fit(x_train, y_train, batch_size=16, verbose=1)
        acc = model.evaluate(x_test, y_test)[1]
        print(acc, max_acc)
        if max_acc < acc:
            max_acc = acc
            model.save("model.h5")


if __name__ == "__main__":
    sys.exit(main())