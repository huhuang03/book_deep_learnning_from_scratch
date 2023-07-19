import argparse
import numpy as np

from .net import ManuallySingleNet
from util.mnist import load_mnist
from . import record

default_train_size = 40000
default_batch_size = 64

# learning_rage = 0.05
learning_rage = 0.1

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
net = ManuallySingleNet()


def insert(index, loss, batch_size=None):
    """
    Parameters
    ----------
    batch_size
    loss: the loss
    index: after {index}th train
    """
    record.insert({
        'index': index,
        'w': net.w,
        'b': net.b,
        'accuracy': net.accuracy(x_test, t_test),
        'loss': loss,
        'batch_size': batch_size,
    })


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', help='batch size', type=int)
    parser.add_argument('-c', help='train count', type=int)
    args = parser.parse_args()
    batch_size = args.b or default_batch_size
    train_count = args.c or default_train_size

    first_train = True
    total_size = x_train.shape[0]
    last = record.get_latest()
    start_index = 0
    if last:
        start_index = last['index']
        net.w = last['w']
        net.b = last['b']
    for i in range(start_index, train_count):
        choice = np.random.choice(total_size, batch_size)
        x = x_train[choice]
        t = t_train[choice]
        before_loss = net.loss(x, t)
        if i == 0:
            insert(0, before_loss)
        print(f'{i} accuracy: ', net.accuracy(x_test, t_test), ', loss: ', before_loss)
        net.gradient(x, t)
        net.w -= (net.dw * learning_rage)
        net.b -= (net.db * learning_rage)
        insert(i + 1, net.loss(x, t), batch_size=batch_size if first_train else None)
        first_train = False


def main():
    train()


if __name__ == '__main__':
    main()
