import numpy as np
import sdl
from .constant import *

# how about do common net?
class TwoNetLayer:
    def __init__(self, input_size, hidden_size, output_size):
        """
        :param input_size:
        :param hidden_size:
        :param output_size:
        """
        self.params = {
            W1: np.random.randn(input_size, hidden_size),
            B1: np.zeros(hidden_size),
            W2: np.random.randn(hidden_size, output_size),
            B2: np.random.randn(output_size)
        }

    def predict(self, x):
        # a1 z1
        a1 = np.dot(x, self.params[W1]) + self.params[B1]
        # print('a1: ', a1)
        z1 = sdl.sigmoid(a1)
        # print('z1: ', z1)
        a2 = np.dot(z1, self.params[W2]) + self.params[B2]
        z2 = sdl.softmax(a2)
        return z2

    def lose(self, x, t):
        # 是不是这个lose不对？
        # print(f'W1: {self.params[W1]}')
        y = self.predict(x)
        # print(f"y: {y}")
        return sdl.cross_entropy_error(y, t)

    def accuracy(self, x, t):
        """
        why have and you?
        :param x:
        :param t:
        :return:
        """
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum((y == t) / float(x.shape[0]))
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.lose(x, t)
        grad = {}
        for key in (W1, B1, W2, B2):
            grad[key] = sdl.numerical_gradient(loss_w, self.params[key])
        return grad