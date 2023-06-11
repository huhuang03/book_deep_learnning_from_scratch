# how to do this?
import numpy as np
from util.mnist import load_mnist
from . import record
import sdl_x


class ManuallySingleNet:
    def __init__(self):
        self.w: np.ndarray = np.random.randn(28 * 28, 10)
        self.b: np.ndarray = np.random.randn(1, 10)
        self.dw = None
        self.db = None

    def forward(self, x: np.ndarray):
        a = x @ self.w + self.b
        y = sdl_x.softmax(sdl_x.sigmoid(a))
        return y

    def accuracy(self, x, t) -> float:
        y = self.forward(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum((y == t) / float(x.shape[0]))
        return accuracy

    def loss(self, x, t) -> float:
        y = self.forward(x)
        return sdl_x.cross_entropy_error(y, t)

    def gradient(self, x, t):
        def f(_):
            return self.loss(x, t)
        self.dw = sdl_x.numerical_gradient(f, self.w)
        self.db = sdl_x.numerical_gradient(f, self.b)


# learning_rage = 0.05
learning_rage = 0.1


(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
net = ManuallySingleNet()

def insert(index, loss):
    """
    :param loss: the loss
    :param index:  after {index}th train
    """
    record.insert({
        'index': index,
        'w': net.w,
        'b': net.b,
        'accuracy': net.accuracy(x_test, t_test),
        'loss': loss
    })

def train():
    batch_size = 16
    total_size = x_train.shape[0]
    last = record.get_latest()
    start_index = 0
    if last:
        start_index = last['index']
        net.w = last['w']
        net.b = last['b']
    for i in range(start_index, 10000):
        choice = np.random.choice(total_size, batch_size)
        x = x_train[choice]
        t = t_train[choice]
        if i == 0:
            insert(0, net.loss(x, t))
        print(f'{i} accuracy: ', net.accuracy(x_test, t_test), ', loss: ', net.loss(x, t))
        net.gradient(x, t)
        net.w -= (net.dw * learning_rage)
        net.b -= (net.db * learning_rage)
        insert(i + 1, net.loss(x, t))


def main():
    train()


if __name__ == '__main__':
    main()
