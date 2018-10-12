import os
import urllib

import cv2
import joblib
import numpy
from keras.engine.saving import load_model

HEIGHT = 128
WIDTH = 128


def get_im(path):
    # Load as grayscale
    img = cv2.imread(path, 0)
    # Reduce size
    resized = cv2.resize(img, (HEIGHT, WIDTH))
    return resized

def get_image_from_url(url):
    img = url_to_image(url)
    resized = cv2.resize(img, (HEIGHT, WIDTH))
    return resized

def url_to_image(url):
    # download the image, convert it to a NumPy array, and then read
    # it into OpenCV format
    resp = urllib.urlopen(url)
    image = numpy.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    # return the image
    return image

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

model = load_model("mode2l.h5")
lb = joblib.load("lb.sav")
print(lb.classes_)
while 1:
    inp = input()
    dataX = []
    binary_image = get_im(inp)
    dataX.append(binary_image)
    dataX = numpy.array(dataX)/255
    dataX = dataX.reshape((-1, HEIGHT, WIDTH, 1))
    print(dataX.shape)
    pred = model.predict(dataX)
    print(pred)