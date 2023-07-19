import os

import cv2
import numpy as np
import torch

from net.custom_dataset import get_custom_dataset
from net.model1 import load_model


y = 0
x = 0


def plt_single_normal_img(index, img, value):
    global y
    global x
    name = f'img_{index}: {value}'
    cv2.imshow(name, img)
    cv2.resizeWindow(name, 500, 200)
    cv2.moveWindow(name, x, y)
    x = x + 500
    if x > 2000:
        y += 200
        x = 0


def plt_single_img(index, img, value):
    img = img * 255
    img = img.reshape((28, 28))
    img = img.astype(np.uint8)
    name = f'img_{index}: {value}'
    cv2.imshow(name, img)
    cv2.resizeWindow(name, 500, 200)


def check_model():
    model = load_model()
    model.eval()
    train_data, test_data = get_custom_dataset(include_mnist=False)
    count = 10
    for index, m in enumerate(test_data):
        img = m[0]
        with torch.no_grad():
            y = model(img)[0].argmax().item()

        img = img.numpy()[0]
        img = img * 255
        img = img.astype(np.uint8)
        plt_single_normal_img(index, img, y)
        index += 1
        if index >= count:
            break
    cv2.waitKey(0)


if __name__ == '__main__':
    check_model()
