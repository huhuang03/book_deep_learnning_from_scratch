from dl_scratch.test.hand_draw_test import handle_draw_test

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

handle_draw_test(lambda img: net.forward(img))