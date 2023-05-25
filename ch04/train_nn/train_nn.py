import sys

import numpy as np
import psutil

from util.mnist import load_mnist
from . import train_recorder
from .. import two_layer_net
from ..constant import *


def try_set_low_priority():
    current_p = psutil.Process()
    if sys.platform == 'win32':
        current_p.nice(0)
    else:
        print('is linux')
        current_p.nice(-20)


def start():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

    # what's this
    train_lose_list = []

    iters_num = 10000
    train_size = x_train.shape[0]
    batch_size = 100
    learning_rage = 0.1

    network = two_layer_net.TwoNetLayer(input_size=784, hidden_size=50, output_size=10)

    for i in range(iters_num):
        record = train_recorder.find_by_index(i)
        if record is not None:
            network.params = {W1: record['w1'], B1: record['b1'], W2: record['w2'], B2: record['b2']}
            continue

        batch_mask = np.random.choice(train_size, batch_size)
        x_batch = x_train[batch_mask]
        t_batch = t_train[batch_mask]

        grad = network.numerical_gradient(x_batch, t_batch)
        for key in (W1, B1, W2, B2):
            network.params[key] -= learning_rage * grad[key]

        record: train_recorder.TrainRecord = {
            'index': i,
            'w1': network.params[W1],
            'b1': network.params[B1],
            'w2': network.params[W2],
            'b2': network.params[B2],
            'loss': network.lose(x_batch, t_batch),
            'accuracy': network.accuracy(x_batch, t_batch)
        }
        train_recorder.insert(record)
        print(f'{i}/{iters_num}: loss: {record["loss"]}, accuracy: {record["accuracy"]}')


if __name__ == '__main__':
    start()
