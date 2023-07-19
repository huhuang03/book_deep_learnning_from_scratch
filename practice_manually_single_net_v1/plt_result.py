import cv2

from .net import ManuallySingleNet
from .record import get_latest
from util.mnist import load_mnist
import numpy as np

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

net = ManuallySingleNet()
weights = get_latest()
net.load(weights['w'], weights['b'])

total_size = x_train.shape[0]
batch_size = 10

choice = np.random.choice(total_size, batch_size)
x = x_train[choice]
t = t_train[choice]


# ok plot this!!
# y = net.forward(x)

def plt_single_img(index, img, value):
    img = img * 255
    img = img.reshape((28, 28))
    img = img.astype(np.uint8)
    name = f'img_{index}: {value}'
    cv2.imshow(name, img)
    cv2.resizeWindow(name, 500, 200)
    cv2.moveWindow(name, index * 500, 0)


for index in range(0, x.shape[0]):
    x_i = x[index]
    y_i = net.forward(x_i)
    plt_single_img(index, x_i, np.argmax(y_i))

cv2.waitKey(0)

# for index in range(0, x.shape[0]):
#     y_i = y[index]
#     x_i = x[index]
#     x_i = x_i * 255
#     x_i = x_i.reshape((28, 28))
#     x_i = x_i.astype(np.uint8)
#     name = f'img_{index}: {np.argmax(y_i)}'
#     cv2.imshow(name, x_i)
#     cv2.resizeWindow(name, 500, 200)
# cv2.waitKey(0)
