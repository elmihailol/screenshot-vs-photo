import configparser
import random

import joblib
import numpy
import os
import sys
import tensorflow as tf

from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Dropout, Flatten
from keras_preprocessing.image import ImageDataGenerator
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from config import IMAGES_PATH, MAXIMUM_IMAGES_PER_CLASS, HEIGHT, WIDTH
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
        dataY = lb.transform(dataY)
        dataY = numpy.array(dataY)
        print(dataX.shape)
        return dataX, dataY

def conv2d_model(output_len = 3):
    input_shape = (HEIGHT, WIDTH, 1)

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(output_len, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model

def main(argv=None):
    print("Read Config")
    ic = ImageContainer(IMAGES_PATH)
    dataX, dataY = ic.get_data_images()
    print("dataX:", len(dataX))
    print("dataY:", len(dataY))
    model = conv2d_model(output_len=3)
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
        h = model.fit_generator(datagen.flow(x_train, y_train, batch_size=4),
                                epochs=1, steps_per_epoch=100,
                                verbose=1)
        acc = model.evaluate(x_test, y_test)[1]
        if max_acc < acc:
            max_acc = acc
            model.save("model.h5")
        print(acc, max_acc)


if __name__ == "__main__":
    sys.exit(main())