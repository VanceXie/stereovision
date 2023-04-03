import imghdr
import os

import cv2
import numpy as np


def cat2images(limg, rimg):
    HEIGHT = limg.shape[0]
    WIDTH = limg.shape[1]
    imgcat = np.zeros((HEIGHT, WIDTH * 2, 3))
    imgcat[:, :WIDTH, :] = limg
    imgcat[:, -WIDTH:, :] = rimg
    for i in range(int(HEIGHT / 32)):
        imgcat[i * 32, :, :] = [0, 0, 255]
    return imgcat


def resize_image(path: str, dsize=(0, 0), scale_x=0.5, scale_y=0.5):
    filetype = ['bmp', 'jpg', 'jpeg', 'png', 'gif']
    files = os.listdir(path)
    for file in files:
        file_path = os.path.join(path, file)
        
        if os.path.isfile(file_path):
            # imghdr 可以用来判断文件是否是图片
            if imghdr.what(file_path) in filetype:
                image = cv2.imread(file_path)
                image_resized = cv2.resize(image, dsize, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join(path, 'left_resized', file), image_resized)
        elif os.path.isdir(file_path):
            pass
