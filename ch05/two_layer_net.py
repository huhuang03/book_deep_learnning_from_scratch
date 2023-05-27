from collections import OrderedDict
import numpy as np
import sdl_x
from ch04.constant import *

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
        self.layers = OrderedDict()
        self.layers['Affine1'] = sdl_x.Affine(self.params[W1], self.params[B1])
        self.layers['Relu'] = sdl_x.Relu()
        self.layers['Affine2'] = sdl_x.Affine(self.params[W2], self.params[B2])
        self.last_layer = sdl_x.SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
        return x

    def lose(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum((y == t) / float(x.shape[0]))
        return accuracy

    def numerical_gradient(self, x, t):
        loss_w = lambda w: self.lose(x, t)
        grad = {}
        for key in (W1, B1, W2, B2):
            grad[key] = sdl_x.numerical_gradient(loss_w, self.params[key])
        return grad

    def gradient(self, x, t):
        self.lose(x, t)
        dout = 1
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {W1: self.layers['Affine1'].dW,
                 B1: self.layers['Affine1'].db,
                 W2: self.layers['Affine2'].dW,
                 B2: self.layers['Affine2'].db}
        return grads