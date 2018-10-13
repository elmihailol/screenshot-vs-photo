import os

from config import MAXIMUM_IMAGES_PER_CLASS
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
                    self.images[key].append([binary_image])
                    counter += 1
                except Exception as e:
                    print(repr(e))

        for key, val in self.images.items():
            print(key, len(val))