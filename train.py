import os

from keras.callbacks import ModelCheckpoint, EarlyStopping

from config import IMAGES_PATH, MAXIMUM_IMAGES_PER_CLASS, HEIGHT, WIDTH, MODEL_TYPE, GPU_USAGE, BATCH_SIZE

if not GPU_USAGE:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import random
from models import image_net_model, conv2d_model
import joblib
import numpy
import os
import sys

from keras.engine.saving import load_model

from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

from helpers import get_im


class ImageContainer:
    def __init__(self, data_path):
        """
        Создает контейнер с изображениями
        :param data_path: dict {class1: path1, class2: path2, ....}
        """
        self.images = {}
        for key, val in data_path.items():
            self.images[key] = []
            print(key, val)
            folder_list = os.listdir(val)
            random.shuffle(folder_list)

            counter = 0
            for file in folder_list:
                if counter % 10 == 0:
                    print(counter,"/", MAXIMUM_IMAGES_PER_CLASS)
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



def main(argv=None):


    print("Read Config")
    ic = ImageContainer(IMAGES_PATH)
    dataX, dataY = ic.get_data_images()
    print("dataX:", len(dataX))
    print("dataY:", len(dataY))
    model = None
    if MODEL_TYPE == "default":
        model = conv2d_model(output_len=3)
    if MODEL_TYPE == "loaded":
        model = load_model("model.h5")
    if MODEL_TYPE == "ImageNet":
        model = image_net_model()
    x_train, x_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2)


    print("start")

    callbacks = [
        EarlyStopping(
            monitor='val_acc',
            patience=10,
            mode='max',
            verbose=1),
        ModelCheckpoint("model.h5",
                        monitor='val_acc',
                        save_best_only=True,
                        mode='max',
                        verbose=1)
    ]

    model.fit(x_train, y_train, batch_size=BATCH_SIZE, verbose=1,
              callbacks=callbacks, validation_data=(x_test, y_test), epochs=100)



if __name__ == "__main__":
    sys.exit(main())