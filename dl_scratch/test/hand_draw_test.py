import os

import cv2
import numpy as np

def handle_draw_test_img(forward):
    root = os.path.join(os.getcwd(), 'recognize_number_django', 'runtime')

    index = 0

    for file in os.listdir(root):
        f_path = os.path.join(root, file)

        x_i = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        y_i = forward(x_i)
        plt_single_normal_img(index, x_i, np.argmax(y_i))
        index += 1
    cv2.waitKey(0)


def handle_draw_test(forward):
    root = os.path.join(os.getcwd(), 'recognize_number_django', 'runtime')

    index = 0

    for file in os.listdir(root):
        f_path = os.path.join(root, file)

        img = cv2.imread(f_path, cv2.IMREAD_GRAYSCALE)
        img = img.flatten()
        x_i = img / 255
        y_i = forward(x_i)
        plt_single_img(index, x_i, np.argmax(y_i))
        index += 1
    cv2.waitKey(0)

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
    cv2.moveWindow(name, index * 500, 0)

