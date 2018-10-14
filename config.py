# Классы и их местоположение на диске
IMAGES_PATH = {
    "screenshot": "C:\data\screenshots",
    "pictures": "C:\data\images",
    "photo": "C:\data\photos"
}

# Максимальное кол-во изображений в обучающей выборке
MAXIMUM_IMAGES_PER_CLASS = 50
BATCH_SIZE = 8
# Тип модели
# default - новая Conv2D
# loaded - model.h5
# ImageNet - ImageNet модель
MODEL_TYPE = "loaded"

# Размер изображения
HEIGHT = 128
WIDTH = 128

# Использовать GPU (Windows only)
GPU_USAGE = True