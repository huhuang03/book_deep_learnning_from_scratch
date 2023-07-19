import cv2
import numpy as np

from util.mnist import load_mnist

# (x_train, t_train), (x_test, t_test) = load_mnist(normalize=False, one_hot_label=True)
#
# random_index = np.random.randint(0, x_train.shape[0])
# random_img = x_train[random_index]
# img = random_img.reshape((28, 28))
# cv2.imshow('img', img)
# cv2.waitKey(0)

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

random_index = np.random.randint(0, x_train.shape[0])
random_img = x_train[random_index]
# random_img = random_img.reshape((28, 28))
# random_img = random_img * 255
# random_img = random_img.astype(np.uint8)
print(random_img)
# cv2.imshow('img1', random_img)
# cv2.waitKey(0)
