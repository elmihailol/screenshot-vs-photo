import cv2

HEIGHT = 128
WIDTH = 128
MAX_IMAGES = 10000
IMAGE_FOLDER = "/home/elmihailol/datasets/images"
PHOTO_FOLDER = "/home/elmihailol/datasets/photos"
SCREENSHOT_FOLDER = "/home/elmihailol/datasets/screenshots"

def get_im(path):
    img = cv2.imread(path, 0)
    resized = cv2.resize(img, (HEIGHT, WIDTH))
    return resized
