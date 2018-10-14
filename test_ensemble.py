import heapq

import joblib
from keras import Model, Input, Sequential
from keras.engine.saving import load_model
from keras.layers import Concatenate, Add, concatenate, Dense
from scipy.misc import imread, imresize

from config import HEIGHT, WIDTH


def predict_class(model, x, lb):
    """
    Предсказывает класс картинки на основе
    :param lb: LabelBinarizer
    :param models: Список моделей
    :param x: картинка
    :return: имя класса
    """

    try:
        pred = model[0].predict(x)[0]

        for i in range(len(model)):
            pred_new = model2.predict(x)[0]
            pred = pred + pred_new
        map_output = {}
        for i in range(len(pred)):
            print(lb.classes_[i], "\t", pred[i])

        max_pred = heapq.nlargest(1, range(len(pred)), pred.take)[0]
        print("Результат:", lb.classes_[max_pred])
        return lb.classes_[max_pred]
    except Exception as e:
        print(repr(e))
        return 0



model1 = load_model("models/model1/model.h5")
model2 = load_model("models/model2/model.h5")

lb = joblib.load("lb.sav")
print(lb.classes_)

while 1:
    try:
        print("\nРасположение изображения:")
        inp = "test/" + input()

        x = imread(inp, mode='L')
        x = imresize(x, (WIDTH, HEIGHT))

        x = x.reshape(1, WIDTH, HEIGHT, 1) / 255

        print(x.shape)
        print(predict_class([model1, model2], x, lb))
    except Exception as e:
        print(repr(e))
