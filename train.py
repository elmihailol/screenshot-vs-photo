import os
import random

import cv2
import joblib
import keras
import numpy
import tensorflow
from keras import Sequential
from keras.engine.saving import load_model
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation
from keras_preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer

from config import IMAGES_PATH
from helpers import get_im

IMAGES_FOLDER_LIST = IMAGES_PATH["picture"]
PHOTO_FOLDER_LIST = IMAGES_PATH["photo"]
SCREENSHOT_FOLDER_LIST = IMAGES_PATH["screenshot"]

IMAGES_DATA = []
PHOTO_DATA = []
SCREENSHOT_DATA = []
print("Получаем обычные картинки...")
for i in range(IMAGES_PATH):
    print(i)
    try:
        binary_image = get_im(IMAGE_FOLDER + "/" + IMAGES_FOLDER_LIST[i])
        IMAGES_DATA.append([binary_image])
    except Exception as e:
        print(repr(e))

    try:
        binary_image = get_im(PHOTO_FOLDER + "/" + PHOTO_FOLDER_LIST[i])
        PHOTO_DATA.append([binary_image])
    except Exception as e:
        print(repr(e))

    try:
        binary_image = get_im(SCREENSHOT_FOLDER + "/" + SCREENSHOT_FOLDER_LIST[i])
        SCREENSHOT_DATA.append([binary_image])
    except Exception as e:
        print(repr(e))
        continue

print("Обычных картинок:", len(IMAGES_DATA))
print("Скриншотов:", len(SCREENSHOT_DATA))
print("Фотографий:", len(PHOTO_DATA))

dataX = []
dataY = []

dataX.extend(IMAGES_DATA)
dataY.extend(["Картинка"] * len(IMAGES_DATA))

dataX.extend(SCREENSHOT_DATA)
dataY.extend(["Скриншот"] * len(SCREENSHOT_DATA))

dataX.extend(PHOTO_DATA)
dataY.extend(["Фотография"] * len(PHOTO_DATA))

mlb = LabelBinarizer()
mlb.fit(dataY)
dataY = mlb.transform(dataY)
print("dataX:", len(dataX))
print("dataY:", len(dataY))

for i in range(len(dataY)):
    print(dataY[i])
dataX = numpy.array(dataX) / 256
dataX = dataX.reshape((-1, HEIGHT, WIDTH, 1))

inputShape = (HEIGHT, WIDTH, 1)
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=inputShape))
model.add(Activation('relu'))
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

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(3, activation='softmax'))

mlb = joblib.load("lb.sav")
model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=['accuracy'])

model = load_model("model.h5")

dataX = dataX.tolist()
dataY = dataY.tolist()
c = list(zip(dataX, dataY))

random.shuffle(c)

dataX, dataY = zip(*c)
train_size = int(len(dataX) * 0.8)

print(dataX[0])
trainX = dataX[:train_size]
testX = dataX[train_size:]

trainY = dataY[:train_size]
testY = dataY[train_size:]

trainX = numpy.array(trainX)
trainY = numpy.array(trainY)
testX = numpy.array(testX)
testY = numpy.array(testY)

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
    h = model.fit_generator(datagen.flow(trainX, trainY, batch_size=128),
                            epochs=1, steps_per_epoch=256,
                            verbose=1)
    acc = model.evaluate(testX, testY)[1]
    if max_acc < acc:
        max_acc = acc
        model.save("model.h5")
    print(acc, max_acc)
